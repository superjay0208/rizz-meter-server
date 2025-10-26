import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
import requests, os, re, math, statistics
from datetime import datetime

OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")

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

app = FastAPI(title="Rizz Meter Server")

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

POSITIVE_WORDS = {"great","awesome","cool","love","amazing","thank you","thanks","wonderful","fantastic","appreciate"}
APPRECIATION_PATTERNS = [r"\b(thanks|thank you|appreciate|that’s great|so glad)\b"]
LAUGHTER_PATTERNS = [r"\b(lol|haha|lmao|rofl|\[laughs\]|(ha){2,})\b"]
BACKCHANNELS = {"yeah","uh-huh","mm-hmm","right","gotcha","i see","ok","okay","mhmm","yup"}
BOUNDARY_PHRASES = [r"\b(not comfortable|don’t want to talk|rather not|can we change the topic|let’s change the topic|no, thanks)\b"]

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def to_0_100(x: float) -> int:
    return int(round(clamp01(x) * 100))

def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())

def contains_any(text: str, patterns: List[str]) -> bool:
    tl = text.lower()
    return any(re.search(p, tl) for p in patterns)

def count_matches(text: str, patterns: List[str]) -> int:
    tl = text.lower()
    return sum(1 for p in patterns if re.search(p, tl))

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
    if segments:
        total_time = segments[-1].end - segments[0].start
    else:
        total_time = 0.0
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

def send_notification(uid: str, title: str, body: str):
    print(f"Attempting to send v2 notification to user: {uid}")
    if not OMI_APP_ID or not OMI_APP_SECRET:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_APP_SECRET is not set.")
        return
    full_message = f"{title}: {body}"
    notification_url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json", "Content-Length": "0"}
    params = {"uid": uid, "message": full_message}
    try:
        resp = requests.post(notification_url, headers=headers, params=params, data="")
        if 200 <= resp.status_code < 300:
            print("✅ Notification sent.")
            print(f"Response: {resp.text}")
        else:
            print(f"❌ Failed notification. Status: {resp.status_code}")
            print(f"Response body: {resp.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def compose_notification(memory_title: str, final: int, metrics: Dict[str, Dict], strengths: List[str], tips: List[str]) -> str:
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

    title_snippet = truncate(memory_title, 60)

    body_multiline = (
        f"Date Rizz: {final}/100 — “{title_snippet}”\n"
        f"{breakdown}\n"
        f"✅ Highlights: {hi}\n"
        f"💡 Try: {im}"
    )

    # Fallback to compact single-line if your push client strips newlines
    if len(body_multiline) > 500:
        body_compact = (
            f"Date Rizz {final}/100 — “{title_snippet}”. "
            f"{breakdown}. "
            f"Highlights: {hi}. "
            f"Try: {im}"
        )
        return body_compact

    return body_multiline

@app.post("/memory_created")
async def analyze_memory(memory: Memory, uid: str):
    print(f"🎉 Analyzing Memory: {memory.structured.title} for user: {uid}")
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
    report_title = "Your Rizz Report is Ready"
    report_body = compose_notification(memory.structured.title, final_score, metrics, strengths, tips)
    send_notification(uid, title=report_title, body=report_body)
    return {"status": "success", "summary": {"title": memory.structured.title, "final_score": final_score, "subscores": {"reciprocity": reciprocity, "attentiveness": attentiveness, "warmth": warmth, "comfort": comfort, "boundary": boundary, "chemistry": chemistry}, "supporting_signals": {"interruptions": interruptions, "backchannels": backchannels, "followups": followups}, "highlights": strengths, "improvements": tips, "generated_at": datetime.utcnow().isoformat() + "Z"}}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Rizz Meter server on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

