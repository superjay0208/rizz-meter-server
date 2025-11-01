import os
import re
import math
import asyncio
import statistics
import requests
import uvicorn
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

# =========================
# Env / constants
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")

# Auto-finalization knobs
IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "180"))   # wall-clock idle -> finalize on next batch
MAX_SEG_GAP_SEC = float(os.environ.get("MAX_SEG_GAP_SEC", "120"))   # audio timeline gap -> finalize on next batch

PID = os.getpid()

# =========================
# Base models (memory_created)
# =========================
class TranscriptSegment(BaseModel):
    text: str
    speaker: str
    start: float
    end: float

class StructuredMemory(BaseModel):
    title: str
    overview: str
    emoji: str

class Memory(BaseModel):
    id: str
    started_at: str
    finished_at: str
    transcript_segments: List[TranscriptSegment]
    structured: StructuredMemory
    apps_response: Optional[List[dict]] = Field(alias="apps_response", default=[])

# =========================
# Real-time payload models
# =========================
class RTIncomingSegment(BaseModel):
    id: Optional[str] = None
    text: str
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    is_user: Optional[bool] = None
    person_id: Optional[str] = None
    start: float
    end: float
    translations: Optional[List[dict]] = None
    speech_profile_processed: Optional[bool] = None

class RTTranscriptBatch(BaseModel):
    segments: List[RTIncomingSegment]
    session_id: Optional[str] = None   # treated as uid fallback
    uid: Optional[str] = None          # accepted too

app = FastAPI(title="Rizz Meter Server")

# =========================
# Optional NLP
# =========================
HF_PIPELINE = None
VADER = None
try:
    from transformers import pipeline
    HF_PIPELINE = pipeline("sentiment-analysis")
except Exception:
    HF_PIPELINE = None

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
except Exception:
    VADER = None

# =========================
# Heuristics/utilities
# =========================
POSITIVE_WORDS = {"great","awesome","cool","love","amazing","thank you","thanks","wonderful","fantastic","appreciate"}
APPRECIATION_PATTERNS = [r"\b(thanks|thank you|appreciate|that’s great|so glad)\b"]
LAUGHTER_PATTERNS = [r"\b(lol|haha|lmao|rofl|\[laughs\]|(ha){2,})\b"]
BACKCHANNELS = {"yeah","uh-huh","mm-hmm","right","gotcha","i see","ok","okay","mhmm","yup"}
BOUNDARY_PHRASES = [r"\b(not comfortable|don’t want to talk|rather not|can we change the topic|let’s change the topic|no, thanks)\b"]

# Start / End markers — broader & case-insensitive
START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)  # matches 'conversation end'/'ends' + typo 'conversaition'

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def to_0_100(x: float) -> int:
    return int(round(clamp01(x) * 100))

def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())

def contains_any(text: str, patterns: List[str]) -> bool:
    tl = text.lower()
    return any(re.search(p, tl) for p in patterns)

def avg(lst: List[float]) -> float:
    return sum(lst)/len(lst) if lst else 0.0

def safe_var(lst: List[float]) -> float:
    return statistics.pvariance(lst) if len(lst) >= 2 else 0.0

# =========================
# Analytics (same as before)
# =========================
def analyze_reciprocity(segments: List[TranscriptSegment]) -> Dict:
    talk_time: Dict[str, float] = {}
    for seg in segments:
        dur = max(0.0, seg.end - seg.start)
        talk_time[seg.speaker] = talk_time.get(seg.speaker, 0.0) + dur
    spk = list(talk_time.keys())
    if len(spk) != 2:
        return {"score": 50, "error": "Not a 2-person conversation", "talk_time": talk_time}
    a, b = spk[0], spk[1]
    tot = talk_time[a] + talk_time[b]
    balance = 0.5 if tot == 0 else talk_time[a]/tot
    reciprocity_score = 1 - min(1.0, abs(balance - 0.5) * 2)
    return {"score": to_0_100(reciprocity_score), "balance": f"{a}: {round(balance*100)}% | {b}: {round((1-balance)*100)}%", "talk_time": talk_time}

def analyze_interruptions(segments: List[TranscriptSegment]) -> Dict:
    interrupts = 0
    swaps = 0
    for i in range(1, len(segments)):
        prev, cur = segments[i-1], segments[i]
        if prev.speaker != cur.speaker:
            swaps += 1
            gap = cur.start - prev.end
            if gap < 0.2:
                interrupts += 1
    rate = 0 if swaps == 0 else interrupts / swaps
    score = 1 - clamp01(rate)
    return {"score": to_0_100(score), "interruptions": interrupts, "speaker_swaps": swaps, "rate": round(rate, 2)}

def analyze_backchannels(segments: List[TranscriptSegment]) -> Dict:
    bc_by_speaker: Dict[str,int] = {}
    for seg in segments:
        dur = seg.end - seg.start
        txt = seg.text.strip().lower()
        if dur <= 1.2 or len(words(txt)) <= 3:
            if txt in BACKCHANNELS or any(w in BACKCHANNELS for w in words(txt)):
                bc_by_speaker[seg.speaker] = bc_by_speaker.get(seg.speaker,0)+1
    total = sum(bc_by_speaker.values())
    score = min(total/5.0, 1.0)
    return {"score": to_0_100(score), "total_backchannels": total, "by_speaker": bc_by_speaker}

def analyze_attentiveness(segments: List[TranscriptSegment]) -> Dict:
    questions: Dict[str,int] = {}
    for seg in segments:
        if seg.text.strip().endswith("?"):
            questions[seg.speaker] = questions.get(seg.speaker, 0) + 1
    total_questions = sum(questions.values())
    total_time = (segments[-1].end - segments[0].start) if segments else 0.0
    target_rate = 1.0 / 30.0
    actual_rate = 0.0 if total_time <= 0 else total_questions / total_time
    score = clamp01(actual_rate / target_rate)
    return {"score": to_0_100(score), "total_questions": total_questions, "breakdown": questions}

def _recent_keywords_by_speaker(segments: List[TranscriptSegment], lookback: int=12) -> Dict[str, set]:
    stop = {"the","a","an","and","or","but","if","to","of","in","on","for","with","it","is","was","be","are","am","that","this","i","you"}
    by_spk: Dict[str, set] = {}
    for seg in segments[-lookback:]:
        toks = [w for w in words(seg.text) if w not in stop and len(w) >= 3]
        key = set(toks[:12])
        by_spk.setdefault(seg.speaker, set()).update(key)
    return by_spk

def analyze_followups(segments: List[TranscriptSegment]) -> Dict:
    if len(segments) < 3:
        return {"score": 50, "followups": 0}
    partner_keywords = _recent_keywords_by_speaker(segments, lookback=12)
    followups = 0
    questions_checked = 0
    def partner_of(spk: str) -> Optional[str]:
        others = {s.speaker for s in segments if s.speaker != spk}
        return next(iter(others)) if others else None
    for seg in segments:
        if not seg.text.strip().endswith("?"):
            continue
        questions_checked += 1
        partner = partner_of(seg.speaker)
        if not partner:
            continue
        keys = partner_keywords.get(partner, set())
        if keys and any(k in words(seg.text) for k in keys):
            followups += 1
    rate = 0 if questions_checked == 0 else followups / questions_checked
    score = clamp01(rate / 0.6)
    return {"score": to_0_100(score), "followups": followups, "questions_checked": questions_checked, "rate": round(rate,2)}

def analyze_sentiment(segments: List[TranscriptSegment]) -> Dict:
    per_seg_scores = []
    pos_tokens = 0
    laughs = 0
    for seg in segments:
        txt = seg.text.strip()
        s = 0.5
        try:
            if HF_PIPELINE:
                res = HF_PIPELINE(txt[:512])[0]
                label = res.get("label","NEUTRAL").upper()
                score = float(res.get("score", 0.5))
                s = score if "POS" in label else (1.0-score if "NEG" in label else 0.5)
            elif VADER:
                vs = VADER.polarity_scores(txt)
                s = (vs["compound"] + 1.0)/2.0
            else:
                wl = txt.lower()
                hits = sum(1 for w in POSITIVE_WORDS if w in wl)
                s = clamp01(0.5 + 0.1*hits)
        except Exception:
            s = 0.5
        per_seg_scores.append((seg.start, s))
        pos_tokens += sum(1 for p in APPRECIATION_PATTERNS if re.search(p, txt.lower()))
        if contains_any(txt, LAUGHTER_PATTERNS):
            laughs += 1
    if not per_seg_scores:
        return {"score": 50}
    t0 = per_seg_scores[0][0]
    xs = [t - t0 for (t, _) in per_seg_scores]
    ys = [y for (_, y) in per_seg_scores]
    if len(xs) >= 2:
        xbar, ybar = avg(xs), avg(ys)
        num = sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))
        den = sum((x-xbar)**2 for x in xs) or 1e-9
        slope = num/den
    else:
        slope = 0.0
    mean_pos = avg(ys)
    warmth = clamp01(0.7*mean_pos + 0.2*clamp01(pos_tokens/3.0) + 0.1*clamp01(laughs/4.0))
    return {"score": to_0_100(warmth), "mean_positivity": round(mean_pos, 3), "slope": round(slope, 4), "appreciation_count": pos_tokens, "laughter_count": laughs}

def analyze_comfort(segments: List[TranscriptSegment]) -> Dict:
    pauses = []
    for i in range(1, len(segments)):
        prev, cur = segments[i-1], segments[i]
        if cur.speaker != prev.speaker:
            gap = cur.start - prev.end
            if 0 < gap < 10.0:
                pauses.append(gap)
    avg_pause = avg(pauses)
    if avg_pause == 0:
        pause_score = 0.6
    else:
        dist = abs(avg_pause - 1.0)
        pause_score = clamp01(1 - dist/0.8)
    rates = []
    for seg in segments:
        dur = max(0.3, seg.end - seg.start)
        w = len(words(seg.text))
        rates.append(w/dur)
    cv = (math.sqrt(safe_var(rates))/avg(rates)) if rates and avg(rates) > 0 else 0.0
    rate_score = clamp01(1 - max(0.0, (cv - 0.2)/(0.5 - 0.2 + 1e-9)))
    score = clamp01(0.6*pause_score + 0.4*rate_score)
    return {"score": to_0_100(score), "average_pause_s": round(avg_pause or 0, 2), "speaking_rate_cv": round(cv, 3)}

def analyze_boundary_respect(segments: List[TranscriptSegment]) -> Dict:
    events = 0
    pushes = 0
    for i, seg in enumerate(segments):
        if contains_any(seg.text, BOUNDARY_PHRASES):
            events += 1
            spk = seg.speaker
            for nxt in segments[i+1:i+4]:
                if nxt.speaker != spk and nxt.text.strip().endswith("?"):
                    pushes += 1
                    break
    if events == 0:
        return {"score": 100, "boundary_events": 0, "pushes_after_no": 0}
    push_rate = pushes / events
    score = clamp01(1 - push_rate)
    return {"score": to_0_100(score), "boundary_events": events, "pushes_after_no": pushes}

def analyze_chemistry(segments: List[TranscriptSegment]) -> Dict:
    q_asked = 0
    q_answered = 0
    for i, seg in enumerate(segments):
        if seg.text.strip().endswith("?"):
            q_asked += 1
            asker = seg.speaker
            for nxt in segments[i+1:i+3]:
                if nxt.speaker != asker and not nxt.text.strip().endswith("?"):
                    q_answered += 1
                    break
    resp_rate = 0 if q_asked == 0 else q_answered/q_asked
    def is_pos_like(t: str) -> bool:
        t = t.lower()
        return contains_any(t, LAUGHTER_PATTERNS) or any(w in t for w in POSITIVE_WORDS)
    sync_events = 0
    sync_hits = 0
    for i, seg in enumerate(segments):
        if is_pos_like(seg.text):
            sync_events += 1
            for nxt in segments[i+1:i+3]:
                if nxt.speaker != seg.speaker and is_pos_like(nxt.text):
                    sync_hits += 1
                    break
    sync_rate = 0 if sync_events == 0 else sync_hits/sync_events
    score = clamp01(0.6*resp_rate + 0.4*sync_rate)
    return {"score": to_0_100(score), "qa_responsiveness": round(resp_rate,2), "positivity_synchrony": round(sync_rate,2)}

WEIGHTS = {"Reciprocity": 0.18, "Attentiveness": 0.18, "Warmth": 0.18, "Comfort": 0.18, "Boundary": 0.14, "Chemistry": 0.10, "Interruptions": 0.0, "Backchannels": 0.0, "FollowUps": 0.0}

def summarize_strengths_and_tips(metrics: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
    strengths, tips = [], []
    core = {k: v["score"] for k,v in metrics.items() if k in ["Reciprocity","Attentiveness","Warmth","Comfort","Boundary","Chemistry"]}
    for k, s in sorted(core.items(), key=lambda kv: kv[1], reverse=True)[:2]:
        if k == "Reciprocity":
            strengths.append("Balanced turn-taking—nice give-and-take.")
        elif k == "Attentiveness":
            strengths.append("Strong curiosity—good questions.")
        elif k == "Warmth":
            strengths.append("Warm tone and positive acknowledgments.")
        elif k == "Comfort":
            strengths.append("Comfortable pacing and pauses.")
        elif k == "Boundary":
            strengths.append("Good boundary respect.")
        elif k == "Chemistry":
            strengths.append("Easy back-and-forth—good ‘chemistry’ cues.")
    lows = sorted(core.items(), key=lambda kv: kv[1])[:2]
    for k, _ in lows:
        if k == "Reciprocity": tips.append("Aim for ~50/50 talk time; invite them in if you’ve led a while.")
        if k == "Attentiveness": tips.append("Ask a follow-up that reuses a detail they shared earlier.")
        if k == "Warmth": tips.append("Add a quick appreciation (e.g., “Thanks for sharing that”).")
        if k == "Comfort": tips.append("Let a ~1s beat after jokes or new topics—don’t rush.")
        if k == "Boundary": tips.append("If they pass on a topic, pivot and check-in before moving on.")
        if k == "Chemistry": tips.append("Answer questions directly, then volley a question back.")
    if "Interruptions" in metrics and metrics["Interruptions"]["score"] < 80:
        tips.append("Avoid cutting in—if excited, use a short back-channel and wait.")
    if "Backchannels" in metrics and metrics["Backchannels"]["score"] < 60:
        tips.append("Sprinkle supportive nods (“mm-hmm”, “I see”) while they talk.")
    if "FollowUps" in metrics and metrics["FollowUps"]["score"] < 60:
        tips.append("Reference their words: “Earlier you mentioned __—tell me more?”")
    return strengths[:2], tips[:2]

def compute_final_score(metrics: Dict[str, Dict]) -> int:
    total = 0.0
    for name, w in WEIGHTS.items():
        if w == 0:
            continue
        s = metrics.get(name, {}).get("score", 0)
        total += w * s
    return int(round(total))

# =========================
# Push helpers
# =========================
def send_notification(uid: str, title: str, body: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting push to uid={uid}")
    if not OMI_APP_ID or not OMI_APP_SECRET:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_APP_SECRET is not set.")
        return
    full_message = f"{title}: {body}"
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json", "Content-Length": "0"}
    params = {"uid": uid, "message": full_message}
    try:
        resp = requests.post(url, headers=headers, params=params, data="")
        if 200 <= resp.status_code < 300:
            print("✅ Notification sent.")
            print(f"Response: {resp.text}")
        else:
            print(f"❌ Failed notification. Status: {resp.status_code}")
            print(f"Response body: {resp.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def compose_notification(title: str, final: int, metrics: Dict[str, Dict], strengths: List[str], tips: List[str]) -> str:
    def truncate(s: str, n: int = 60) -> str:
        return (s[:n].rstrip() + "…") if len(s) > n else s
    br = {
        "R": metrics["Reciprocity"]["score"],
        "A": metrics["Attentiveness"]["score"],
        "W": metrics["Warmth"]["score"],
        "C": metrics["Comfort"]["score"],
        "B": metrics["Boundary"]["score"],
        "Ch": metrics["Chemistry"]["score"],
    }
    breakdown = f"R {br['R']} · A {br['A']} · W {br['W']} · C {br['C']} · B {br['B']} · Ch {br['Ch']}"
    hi = " • ".join(strengths) if strengths else "Nice effort!"
    im = " • ".join(tips) if tips else "Keep doing what felt natural."
    tsn = truncate(title or "Conversation", 60)
    body = (
        f"Date Rizz: {final}/100 — “{tsn}”\n"
        f"{breakdown}\n"
        f"✅ Highlights: {hi}\n"
        f"💡 Try: {im}"
    )
    return body if len(body) <= 500 else f"Date Rizz {final}/100 — “{tsn}”. {breakdown}. Highlights: {hi}. Try: {im}"

# =========================
# Per-uid state
# =========================
class ConvState:
    def __init__(self):
        self.active: bool = False
        self.buffer: List[TranscriptSegment] = []
        self.title: Optional[str] = None
        self.last_uid: Optional[str] = None
        self.last_wall_ts: float = 0.0    # wall-clock timestamp of last activity
        self.last_seg_end: float = 0.0    # last audio timeline end
        self.lock = asyncio.Lock()

    def touch_wall(self):
        self.last_wall_ts = datetime.now(timezone.utc).timestamp()

CONVS: Dict[str, ConvState] = {}
CONVS_LOCK = asyncio.Lock()

def _normalize_speaker(seg: RTIncomingSegment) -> str:
    if seg.is_user is True:
        return "You"
    if seg.is_user is False:
        return "Partner"
    if seg.speaker is not None:
        return seg.speaker
    if seg.speaker_id is not None:
        return f"SPEAKER_{seg.speaker_id}"
    return "Unknown"

def _rt_to_internal(seg: RTIncomingSegment) -> TranscriptSegment:
    return TranscriptSegment(
        text=seg.text or "",
        speaker=_normalize_speaker(seg),
        start=float(seg.start),
        end=float(seg.end),
    )

def _is_start_marker(txt: str) -> bool:
    return bool(START_RE.search(txt or ""))

def _is_end_marker(txt: str) -> bool:
    return bool(END_RE.search(txt or ""))

async def _get_state(uid: str) -> ConvState:
    async with CONVS_LOCK:
        if uid not in CONVS:
            CONVS[uid] = ConvState()
        return CONVS[uid]

async def _finalize_and_analyze(uid: str) -> Dict:
    state = await _get_state(uid)
    async with state.lock:
        segs = list(state.buffer)
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔔 Finalizing uid={uid} with {len(segs)} segments")

        def non_marker_texts():
            for s in segs:
                if not _is_start_marker(s.text) and not _is_end_marker(s.text):
                    t = s.text.strip()
                    if t:
                        yield t
        title = state.title or next((t[:60] for t in non_marker_texts()), "Conversation")
        clean_segments = [s for s in segs if not _is_start_marker(s.text) and not _is_end_marker(s.text)]

        if len(clean_segments) < 2:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⚠️ Not enough segments to analyze (uid={uid}).")
            summary = {"status": "error", "message": "Not enough segments to analyze."}
        else:
            # analytics
            reciprocity = analyze_reciprocity(clean_segments)
            interruptions = analyze_interruptions(clean_segments)
            backchannels = analyze_backchannels(clean_segments)
            attentiveness = analyze_attentiveness(clean_segments)
            followups = analyze_followups(clean_segments)
            warmth = analyze_sentiment(clean_segments)
            comfort = analyze_comfort(clean_segments)
            boundary = analyze_boundary_respect(clean_segments)
            chemistry = analyze_chemistry(clean_segments)
            metrics = {
                "Reciprocity": reciprocity, "Interruptions": interruptions, "Backchannels": backchannels,
                "Attentiveness": attentiveness, "FollowUps": followups, "Warmth": warmth,
                "Comfort": comfort, "Boundary": boundary, "Chemistry": chemistry
            }
            final_score = compute_final_score(metrics)
            strengths, tips = summarize_strengths_and_tips(metrics)
            body = compose_notification(title, final_score, metrics, strengths, tips)
            push_uid = state.last_uid or uid
            if push_uid:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 📣 Pushing to uid='{push_uid}', score={final_score}")
                send_notification(push_uid, title="Your Rizz Report is Ready", body=body)
            else:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔕 No uid to push")

            summary = {
                "status": "success",
                "summary": {
                    "title": title,
                    "final_score": final_score,
                    "subscores": {
                        "reciprocity": reciprocity, "attentiveness": attentiveness, "warmth": warmth,
                        "comfort": comfort, "boundary": boundary, "chemistry": chemistry
                    },
                    "supporting_signals": {
                        "interruptions": interruptions, "backchannels": backchannels, "followups": followups
                    },
                    "highlights": strengths,
                    "improvements": tips,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            }

        # reset state
        state.active = False
        state.buffer = []
        state.title = None
        state.last_uid = None
        state.touch_wall()
        state.last_seg_end = 0.0
        return summary

# =========================
# Endpoints
# =========================
@app.get("/")
async def health():
    return {"status": "ok", "omi_creds_loaded": bool(OMI_APP_ID and OMI_APP_SECRET), "pid": PID}

@app.post("/transcript_processed")
async def transcript_processed(
    batch: RTTranscriptBatch,
    uid: Optional[str] = Query(None),
    force_start: Optional[int] = Query(0),
    force_end: Optional[int] = Query(0),
):
    """
    Real-time ingest.
    - uid read from: body.uid -> query ?uid= -> session_id
    - Start on text 'conversation starts' (case-insensitive)
      OR if ?force_start=1 is passed.
    - End on 'conversation end'/'ends'/'conversaition ends'
      OR if ?force_end=1 is passed.
    - Auto-finalize if either:
        * wall-clock idle > IDLE_TIMEOUT_SEC (finalize on next batch), or
        * audio timeline gap > MAX_SEG_GAP_SEC (finalize on next batch).
    """
    if not batch.segments and not force_end and not force_start:
        return {"status": "ignored", "reason": "no_segments", "pid": PID}

    effective_uid = batch.uid or uid or batch.session_id
    if not effective_uid:
        return {"status": "ignored", "reason": "missing_uid(session_id/body/query)", "pid": PID}

    state = await _get_state(effective_uid)
    state.last_uid = effective_uid
    source = "body.uid" if batch.uid else ("query.uid" if uid else "session_id")
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔗 uid set from {source}: {effective_uid}")

    # Detect markers in this batch
    has_start = any(_is_start_marker(s.text) for s in (batch.segments or []))
    has_end   = any(_is_end_marker(s.text) for s in (batch.segments or []))

    async with state.lock:
        now_ts = datetime.now(timezone.utc).timestamp()

        # Soft auto-finalize if previously active and idle too long (handled on arrival of any new batch)
        if state.active and state.last_wall_ts and (now_ts - state.last_wall_ts > IDLE_TIMEOUT_SEC) and state.buffer:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏰ Idle timeout (wall) for uid={effective_uid}. Auto-finalizing previous convo.")
            await _finalize_and_analyze(effective_uid)

        # Manual overrides
        if force_end:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🛑 force_end received for uid={effective_uid}")
            return await _finalize_and_analyze(effective_uid)

        if has_start or force_start:
            if state.active and state.buffer:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔄 start while active (uid={effective_uid}) — auto-finalizing previous.")
                await _finalize_and_analyze(effective_uid)
            state.active = True
            state.buffer = []
            state.title = None
            state.last_seg_end = 0.0
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟢 conversation starts — buffering begins (uid={effective_uid})")

        if not state.active:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏸️ Batch ignored: conversation not started yet. (uid={effective_uid})")
            state.touch_wall()
            return {"status": "ignored", "reason": "not_started", "pid": PID}

        # Buffer incoming segments and check audio timeline gaps
        audio_gap_trigger = False
        for seg in batch.segments or []:
            internal = _rt_to_internal(seg)
            # conversation title from first non-marker text
            if state.title is None and internal.text.strip() and not _is_start_marker(internal.text) and not _is_end_marker(internal.text):
                state.title = internal.text.strip()[:60]

            # detect audio gap from last_seg_end
            if state.last_seg_end and (internal.end - state.last_seg_end) > MAX_SEG_GAP_SEC:
                audio_gap_trigger = True
            state.last_seg_end = max(state.last_seg_end, internal.end)

            state.buffer.append(internal)

        state.touch_wall()

        if has_end:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 end marker detected — finalizing (uid={effective_uid})")
            return await _finalize_and_analyze(effective_uid)

        if audio_gap_trigger and state.buffer:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏰ Large audio gap detected (> {MAX_SEG_GAP_SEC}s). Auto-finalizing (uid={effective_uid}).")
            return await _finalize_and_analyze(effective_uid)

        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔄 Buffering: total segments={len(state.buffer)} (uid={effective_uid})")
        return {"status": "buffering", "segments_buffered": len(state.buffer), "will_push_on_end": True, "pid": PID}

# Manual end endpoint (handy if markers were missed)
@app.post("/conversation/end")
async def conversation_end(uid: str = Query(...)):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🛑 /conversation/end called for uid={uid}")
    return await _finalize_and_analyze(uid)

# Back-compat endpoint (memory_created)
@app.post("/memory_created")
async def analyze_memory(memory: Memory, uid: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🎉 Analyzing Memory: {memory.structured.title} for uid: {uid}")
    segments = memory.transcript_segments
    if len(segments) < 2:
        return {"status": "error", "message": "Not enough segments."}
    reciprocity = analyze_reciprocity(segments)
    interruptions = analyze_interruptions(segments)
    backchannels = analyze_backchannels(segments)
    attentiveness = analyze_attentiveness(segments)
    followups = analyze_followups(segments)
    warmth = analyze_sentiment(segments)
    comfort = analyze_comfort(segments)
    boundary = analyze_boundary_respect(segments)
    chemistry = analyze_chemistry(segments)
    metrics = {"Reciprocity": reciprocity, "Interruptions": interruptions, "Backchannels": backchannels, "Attentiveness": attentiveness, "FollowUps": followups, "Warmth": warmth, "Comfort": comfort, "Boundary": boundary, "Chemistry": chemistry}
    final_score = compute_final_score(metrics)
    strengths, tips = summarize_strengths_and_tips(metrics)
    body = compose_notification(memory.structured.title, final_score, metrics, strengths, tips)
    send_notification(uid, title="Your Rizz Report is Ready", body=body)
    return {"status": "success", "summary": {"title": memory.structured.title, "final_score": final_score, "subscores": {"reciprocity": reciprocity, "attentiveness": attentiveness, "warmth": warmth, "comfort": comfort, "boundary": boundary, "chemistry": chemistry}, "supporting_signals": {"interruptions": interruptions, "backchannels": backchannels, "followups": followups}, "highlights": strengths, "improvements": tips, "generated_at": datetime.now(timezone.utc).isoformat()}}

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Rizz Meter server on http://0.0.0.0:{port} (pid={PID})")
    # Keep single worker in prod to avoid split memory problems, or persist state to Redis/DB if you need >1.
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
