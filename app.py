import os
import re
import math
import asyncio
import statistics
import uvicorn
import json
import httpx   # <-- ADDED
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple, Any
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import time

# =========================
# Env / constants
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")

# DeepSeek
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Auto-finalization knobs
IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "180"))
MAX_SEG_GAP_SEC = float(os.environ.get("MAX_SEG_GAP_SEC", "120"))
PID = os.getpid()


url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
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
    session_id: Optional[str] = None
    uid: Optional[str] = None

# =========================
# App lifespan (HTTP Client) <-- MODIFIED
# =========================

# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Server starting up...")
    # Initialize the client with a default timeout
    http_client = httpx.AsyncClient(timeout=45.0)
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 👋 Server shutting down...")

app = FastAPI(title="Rizz Meter Server", lifespan=lifespan)

# =========================
# Heuristics/utilities
# =========================
# ... (All synchronous helper functions like clamp01, words, analyze_reciprocity, etc.
# ...  remain exactly the same. They are not shown here for brevity but are
# ...  assumed to be present from your original code.)
POSITIVE_WORDS = {
    "great","awesome","cool","love","amazing","thank you","thanks","wonderful","fantastic","appreciate"
}
APPRECIATION_PATTERNS = [r"\b(thanks|thank you|appreciate|that’s great|so glad)\b"]
LAUGHTER_PATTERNS = [r"\b(lol|haha|lmao|rofl|\[laughs\]|(ha){2,})\b"]
BACKCHANNELS = {"yeah","uh-huh","mm-hmm","right","gotcha","i see","ok","okay","mhmm","yup"}
BOUNDARY_PHRASES = [r"\b(not comfortable|don’t want to talk|rather not|can we change the topic|let’s change the topic|no, thanks)\b"]

START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)

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
    return {
        "score": to_0_100(reciprocity_score),
        "balance": f"{a}: {round(balance*100)}% | {b}: {round((1-balance)*100)}%",
        "talk_time": talk_time
    }

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
    per_seg_scores: List[Tuple[float, float]] = []
    pos_tokens = 0
    laughs = 0
    for seg in segments:
        txt = seg.text.strip()
        wl = txt.lower()
        hits = sum(1 for w in POSITIVE_WORDS if w in wl)
        s = clamp01(0.5 + 0.1 * hits)
        per_seg_scores.append((seg.start, s))
        pos_tokens += sum(1 for p in APPRECIATION_PATTERNS if re.search(p, wl))
        if contains_any(txt, LAUGHTER_PATTERNS):
            laughs += 1
    if not per_seg_scores:
        return {"score": 50}
    t0 = per_seg_scores[0][0]
    xs = [t - t0 for (t, _) in per_seg_scores]
    ys = [y for (_, y) in per_seg_scores]
    if len(xs) >= 2:
        xbar, ybar = avg(xs), avg(ys)
        num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        den = sum((x - xbar) ** 2 for x in xs) or 1e-9
        slope = num / den
    else:
        slope = 0.0
    mean_pos = avg(ys)
    warmth = clamp01(0.7 * mean_pos + 0.2 * clamp01(pos_tokens / 3.0) + 0.1 * clamp01(laughs / 4.0))
    return {
        "score": to_0_100(warmth),
        "mean_positivity": round(mean_pos, 3),
        "slope": round(slope, 4),
        "appreciation_count": pos_tokens,
        "laughter_count": laughs,
    }

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

WEIGHTS = {
    "Reciprocity": 0.18, "Attentiveness": 0.18, "Warmth": 0.18, "Comfort": 0.18, "Boundary": 0.14, "Chemistry": 0.10,
    "Interruptions": 0.0, "Backchannels": 0.0, "FollowUps": 0.0
}

def compute_final_score(metrics: Dict[str, Dict]) -> int:
    total = 0.0
    for name, w in WEIGHTS.items():
        if w == 0:
            continue
        s = metrics.get(name, {}).get("score", 0)
        total += w * s
    return int(round(total))

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
# =========================
# DeepSeek integration
# =========================
def prepare_transcript_for_llm(segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
    return [
        {
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "speaker": s.speaker,
            "text": s.text.strip(),
        }
        for s in segments
        if s.text and s.text.strip()
    ]

DEEPSEEK_SYSTEM_PROMPT = """You are an expert social-conversation analyst for post-date debriefs.
Your job: read the transcript and return a STRUCTURED JSON assessment focused on dating-relevant signals:
- Reciprocity & turn-taking (talk-time balance, interruptions, back-channels)
- Attentiveness (question rate, follow-ups referencing prior details)
- Warmth (sentiment trajectory, acknowledgments, appreciation)
- Comfort & pacing (speaking-rate stability, pause tolerance, laughter)
- Boundary respect (honoring avoidant topics, not pushing after “no”)
Compute normalized sub-scores (0–100) for: Reciprocity, Attentiveness, Warmth, Comfort, Boundary, Chemistry.
Aggregate to a final 0–100 “Date Rizz” score. Provide 2–3 highlights and 2–3 improvement tips.
Provide a short “highlights_reel” with 2–5 concrete moments (timestamp + quote + why it matters).
Provide 3–5 “suggested_prompts” to use next time.
Return ONLY JSON following the exact schema. No extra commentary."""

def deepseek_user_prompt(transcript_json: List[Dict[str, Any]], title_hint: Optional[str]) -> str:
    schema = {
        "title": "string (<= 80 chars; use first meaningful utterance if not provided)",
        "final_score": "int 0-100",
        "subscores": {
            "reciprocity": {"score": "int 0-100", "balance": "string", "interruptions": "int", "backchannels": "int"},
            "attentiveness": {"score": "int 0-100", "question_rate_per_min": "float", "followups": "int", "examples": ["string", "..."]},
            "warmth": {"score": "int 0-100", "sentiment_trend": "one of ['up','down','flat']", "positives": "int", "examples": ["string", "..."]},
            "comfort": {"score": "int 0-100", "avg_pause_s": "float", "speaking_rate_cv": "float", "laughter_count": "int"},
            "boundary": {"score": "int 0-100", "events": "int", "pushes_after_no": "int", "examples": ["string", "..."]},
            "chemistry": {"score": "int 0-100", "qa_responsiveness": "float 0-1", "positivity_synchrony": "float 0-1"}
        },
        "highlights": ["string", "..."],
        "improvements": ["string", "..."],
        "highlights_reel": [
            {"timestamp_s": "float", "text": "string", "why": "string"}
        ],
        "suggested_prompts": ["string", "..."],
        "generated_at": "ISO-8601 datetime string in UTC"
    }
    rubric = {
        "normalization": "Consider total duration and #turns; avoid penalizing short talks.",
        "aggregation": "Weighting guidance (can adapt): R=0.18, A=0.18, W=0.18, C=0.18, B=0.14, Ch=0.10. Clamp 0–100.",
        "definitions": {
            "followup": "Question referencing a partner detail or keyword from earlier.",
            "backchannel": "Brief listener cue (mm-hmm, yeah, right).",
            "boundary": "Respect for topic declines or explicit 'no'."
        }
    }
    return json.dumps({
        "title_hint": (title_hint or "")[:80],
        "scoring_rubric": rubric,
        "required_schema": schema,
        "transcript": transcript_json
    })

# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
async def call_deepseek(messages, temperature=0.2, max_tokens=1500) -> Optional[dict]:
    ...
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    for attempt in range(3):
        try:
            resp = await http_client.post(url, headers=headers, json=payload, timeout=200.0)
        except httpx.ReadTimeout:
            print(f"❌ DeepSeek timeout (attempt {attempt+1})")
            continue
        except Exception as e:
            print(f"❌ DeepSeek network (attempt {attempt+1}): {e}")
            await asyncio.sleep(1.5 * (attempt + 1))
            continue

        if resp.status_code // 100 != 2:
            print(f"❌ DeepSeek HTTP {resp.status_code} (attempt {attempt+1}): {resp.text[:400]}")
            if resp.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return None

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)  # you can keep your wrapper if you want to extract fenced JSON
        except Exception as e:
            print(f"❌ Parse error: {e}. Preview: {resp.text[:400]}")
            return None

    print("❌ DeepSeek: retries exhausted.")
    return None


@app.get("/deepseek/ping")
async def deepseek_ping():
    msgs = [
        {"role":"system","content":"Return ONLY {\"ok\":true,\"ts\":\"<utc>\"}."},
        {"role":"user","content":"Respond with ok=true and current UTC timestamp."}
    ]
    out = await call_deepseek(msgs, temperature=0.0, max_tokens=64)
    return {"bridge_ok": bool(out), "raw": out}
    
    # --- End of New Robust Error Handling ---

def transform_llm_to_metrics(llm: dict, fallback_segments: List[TranscriptSegment]) -> Tuple[Dict[str, Dict], int, List[str], List[str], Dict[str, Any]]:
    subs = llm.get("subscores", {})
    def pull(name, default=50):
        return int(subs.get(name, {}).get("score", default))

    metrics = {
        "Reciprocity": {"score": pull("reciprocity"), **{k:v for k,v in subs.get("reciprocity", {}).items() if k != "score"}},
        "Attentiveness": {"score": pull("attentiveness"), **{k:v for k,v in subs.get("attentiveness", {}).items() if k != "score"}},
        "Warmth": {"score": pull("warmth"), **{k:v for k,v in subs.get("warmth", {}).items() if k != "score"}},
        "Comfort": {"score": pull("comfort"), **{k:v for k,v in subs.get("comfort", {}).items() if k != "score"}},
        "Boundary": {"score": pull("boundary"), **{k:v for k,v in subs.get("boundary", {}).items() if k != "score"}},
        "Chemistry": {"score": pull("chemistry"), **{k:v for k,v in subs.get("chemistry", {}).items() if k != "score"}},
        "Interruptions": {"score": 100},
        "Backchannels": {"score": 100},
        "FollowUps": {"score": 100},
    }

    final_score = int(llm.get("final_score", compute_final_score(metrics)))
    highlights = list(llm.get("highlights", []))[:3]
    improvements = list(llm.get("improvements", []))[:3]
    extras = {
        "highlights_reel": llm.get("highlights_reel", []),
        "suggested_prompts": llm.get("suggested_prompts", []),
        "generated_at": llm.get("generated_at", datetime.now(timezone.utc).isoformat())
    }
    return metrics, final_score, highlights, improvements, extras

# =========================
# Push helpers
# =========================
# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
async def send_notification(uid: str, title: str, body: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting push to uid={uid}")
    if not OMI_APP_ID or not OMI_APP_SECRET:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_APP_SECRET is not set.")
        return
    if not http_client:
        print("❌ HTTP Client not initialized; skipping notification.")
        return
        
    full_message = f"{title}: {body}"
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json", "Content-Length": "0"}
    params = {"uid": uid, "message": full_message}
    try:
        # <<< MODIFIED: Use await and httpx client >>>
        resp = await http_client.post(url, headers=headers, params=params, data="")
        
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

# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
async def save_text_as_memory(uid: str, text_content: str):
    """
    Saves a plain text string as a new memory in Omi.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting to save memory for uid={uid}")
    if not OMI_APP_ID or not OMI_APP_SECRET:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_APP_SECRET is not set. Cannot save memory.")
        return
    if not http_client:
        print("❌ HTTP Client not initialized; skipping memory save.")
        return

    url = "https://api.omi.me/v2/memories"
    headers = {
        "Authorization": f"Bearer {OMI_APP_SECRET}",
        "Content-Type": "application/json"
    }
    params = {"uid": uid}
    payload = {
        "text": text_content,
        "app_id": OMI_APP_ID
    }

    try:
        # <<< MODIFIED: Use await and httpx client (with a specific 15s timeout) >>>
        resp = await http_client.post(url, headers=headers, params=params, json=payload, timeout=15.0)
        
        if 200 <= resp.status_code < 300:
            print(f"✅ Memory saved successfully. ID: {resp.json().get('id')}")
        else:
            print(f"❌ Failed to save memory. Status: {resp.status_code}")
            print(f"Response body: {resp.text}")
    except Exception as e:
        print(f"Error saving memory: {e}")

# =========================
# Per-uid state
# =========================
class ConvState:
    def __init__(self):
        self.active: bool = False
        self.buffer: List[TranscriptSegment] = []
        self.title: Optional[str] = None
        self.last_uid: Optional[str] = None
        self.last_wall_ts: float = 0.0
        self.last_seg_end: float = 0.0
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

# =========================
# Finalization using LLM first (fallback to heuristics)
# =========================
# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
async def llm_analyze(segments: List[TranscriptSegment], title_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    if not segments:
        return None
    transcript_json = prepare_transcript_for_llm(segments)
    messages = [
        {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
        {"role": "user", "content": deepseek_user_prompt(transcript_json, title_hint)}
    ]
    # <<< MODIFIED: Use await >>>
    return await call_deepseek(messages)

def fallback_analyze(clean_segments: List[TranscriptSegment]) -> Dict[str, Any]:
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
    return {
        "status": "success",
        "summary": {
            "title": "Conversation",
            "final_score": final_score,
            "subscores": {
                "reciprocity": reciprocity, "attentiveness": attentiveness, "warmth": warmth,
                "comfort": comfort, "boundary": boundary, "chemistry": chemistry
            },
            "supporting_signals": {"interruptions": interruptions, "backchannels": backchannels, "followups": followups},
            "highlights": strengths,
            "improvements": tips,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "highlights_reel": [],
            "suggested_prompts": []
        }
    }

# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
async def _finalize_and_analyze(uid: str) -> Dict:
    state = await _get_state(uid)
    async with state.lock:
        # <<< MODIFIED: Use await >>>
        return await _finalize_and_analyze_UNLOCKED(state, uid)

# <<< MODIFIED: CONVERTED TO ASYNC DEF >>>
async def _finalize_and_analyze_UNLOCKED(state: ConvState, uid: str) -> Dict:
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
        # <<< MODIFIED: Use await >>>
        llm = await llm_analyze(clean_segments, title)
        
        if llm:
            metrics, final_score, strengths, tips, extras = transform_llm_to_metrics(llm, clean_segments)
            body = compose_notification(title, final_score, metrics, strengths, tips)
            push_uid = state.last_uid or uid
            if push_uid:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 📣 Pushing (LLM) to uid='{push_uid}', score={final_score}")
                # <<< MODIFIED: Use await >>>
                await send_notification(push_uid, title="Your Rizz Report is Ready", body=body)
                await save_text_as_memory(push_uid, body)
            summary = {
                "status": "success",
                "summary": {
                    "title": title,
                    "final_score": final_score,
                    "subscores": {
                        "reciprocity": metrics["Reciprocity"],
                        "attentiveness": metrics["Attentiveness"],
                        "warmth": metrics["Warmth"],
                        "comfort": metrics["Comfort"],
                        "boundary": metrics["Boundary"],
                        "chemistry": metrics["Chemistry"],
                    },
                    "supporting_signals": {
                        "interruptions": metrics.get("Interruptions", {}),
                        "backchannels": metrics.get("Backchannels", {}),
                        "followups": metrics.get("FollowUps", {}),
                    },
                    "highlights": strengths,
                    "improvements": tips,
                    "generated_at": extras.get("generated_at", datetime.now(timezone.utc).isoformat()),
                    "highlights_reel": extras.get("highlights_reel", []),
                    "suggested_prompts": extras.get("suggested_prompts", [])
                }
            }
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⚠️ LLM unavailable — using fallback heuristics.")
            summary = fallback_analyze(clean_segments)

    # Reset state
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
    return {
        "status": "ok",
        "omi_creds_loaded": bool(OMI_APP_ID and OMI_APP_SECRET),
        "deepseek_ready": bool(DEEPSEEK_API_KEY),
        "pid": PID
    }

@app.post("/transcript_processed")
async def transcript_processed(
    batch: RTTranscriptBatch,
    uid: Optional[str] = Query(None),
    force_start: Optional[int] = Query(0),
    force_end: Optional[int] = Query(0),
):
    if not batch.segments and not force_end and not force_start:
        return {"status": "ignored", "reason": "no_segments", "pid": PID}

    effective_uid = batch.uid or uid or batch.session_id
    if not effective_uid:
        return {"status": "ignored", "reason": "missing_uid(session_id/body/query)", "pid": PID}

    state = await _get_state(effective_uid)
    state.last_uid = effective_uid
    source = "body.uid" if batch.uid else ("query.uid" if uid else "session_id")
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔗 uid set from {source}: {effective_uid}")

    has_start = any(_is_start_marker(s.text) for s in (batch.segments or []))
    has_end   = any(_is_end_marker(s.text) for s in (batch.segments or []))

    async with state.lock:
        now_ts = datetime.now(timezone.utc).timestamp()

        if state.active and state.last_wall_ts and (now_ts - state.last_wall_ts > IDLE_TIMEOUT_SEC) and state.buffer:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏰ Idle timeout (wall) for uid={effective_uid}. Auto-finalizing previous convo.")
            # <<< MODIFIED: Use await >>>
            await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        if force_end:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🛑 force_end received for uid={effective_uid}")
            # <<< MODIFIED: Use await >>>
            return await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        if has_start or force_start:
            if state.active and state.buffer:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔄 start while active (uid={effective_uid}) — auto-finalizing previous.")
                # <<< MODIFIED: Use await >>>
                await _finalize_and_analyze_UNLOCKED(state, effective_uid)
            state.active = True
            state.buffer = []
            state.title = None
            state.last_seg_end = 0.0
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟢 conversation starts — buffering begins (uid={effective_uid})")

        if not state.active:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏸️ Batch ignored: conversation not started yet. (uid={effective_uid})")
            state.touch_wall()
            return {"status": "ignored", "reason": "not_started", "pid": PID}

        audio_gap_trigger = False
        for seg in batch.segments or []:
            internal = _rt_to_internal(seg)
            if state.title is None and internal.text.strip() and not _is_start_marker(internal.text) and not _is_end_marker(internal.text):
                state.title = internal.text.strip()[:60]

            if state.last_seg_end and (internal.end - state.last_seg_end) > MAX_SEG_GAP_SEC:
                audio_gap_trigger = True
            state.last_seg_end = max(state.last_seg_end, internal.end)
            state.buffer.append(internal)

        state.touch_wall()

        if has_end:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 end marker detected — finalizing (uid={effective_uid})")
            # <<< MODIFIED: Use await >>>
            return await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        if audio_gap_trigger and state.buffer:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏰ Large audio gap detected (> {MAX_SEG_GAP_SEC}s). Auto-finalizing (uid={effective_uid}).")
            # <<< MODIFIED: Use await >>>
            return await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔄 Buffering: total segments={len(state.buffer)} (uid={effective_uid})")
        return {"status": "buffering", "segments_buffered": len(state.buffer), "will_push_on_end": True, "pid": PID}

@app.post("/conversation/end")
async def conversation_end(uid: str = Query(...)):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🛑 /conversation/end called for uid={uid}")
    # <<< MODIFIED: Use await >>>
    return await _finalize_and_analyze(uid)

@app.post("/memory_created")
async def analyze_memory(memory: Memory, uid: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🎉 Analyzing Memory: {memory.structured.title} for uid: {uid}")
    segments = memory.transcript_segments
    if len(segments) < 2:
        return {"status": "error", "message": "Not enough segments."}

    # <<< MODIFIED: Use await >>>
    llm = await llm_analyze(segments, memory.structured.title)
    
    if llm:
        metrics, final_score, strengths, tips, extras = transform_llm_to_metrics(llm, segments)
        body = compose_notification(memory.structured.title, final_score, metrics, strengths, tips)
        
        # <<< MODIFIED: Use await >>>
        await send_notification(uid, title="Your Rizz Report is Ready", body=body)
        await save_text_as_memory(uid, body)
        
        return {"status": "success", "summary": {
            "title": memory.structured.title,
            "final_score": final_score,
            "subscores": {
                "reciprocity": metrics["Reciprocity"], "attentiveness": metrics["Attentiveness"], "warmth": metrics["Warmth"],
                "comfort": metrics["Comfort"], "boundary": metrics["Boundary"], "chemistry": metrics["Chemistry"]
            },
            "supporting_signals": {
                "interruptions": metrics.get("Interruptions", {}), "backchannels": metrics.get("Backchannels", {}), "followups": metrics.get("FollowUps", {})
            },
            "highlights": strengths, "improvements": tips,
            "generated_at": extras.get("generated_at", datetime.now(timezone.utc).isoformat()),
            "highlights_reel": extras.get("highlights_reel", []),
            "suggested_prompts": extras.get("suggested_prompts", [])
        }}

    # Fallback if LLM unavailable
    fb = fallback_analyze(segments)
    
    notification_body = compose_notification(
        fb["summary"]["title"], fb["summary"]["final_score"],
        {
            "Reciprocity": fb["summary"]["subscores"]["reciprocity"],
            "Attentiveness": fb["summary"]["subscores"]["attentiveness"],
            "Warmth": fb["summary"]["subscores"]["warmth"],
            "Comfort": fb["summary"]["subscores"]["comfort"],
            "Boundary": fb["summary"]["subscores"]["boundary"],
            "Chemistry": fb["summary"]["subscores"]["chemistry"],
        },
        fb["summary"]["highlights"],
        fb["summary"]["improvements"]
    )
    
    # <<< MODIFIED: Use await >>>
    await send_notification(uid, title="Your Rizz Report is Ready", body=notification_body)
    await save_text_as_memory(uid, notification_body)
    
    return fb

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Rizz Meter server on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)



