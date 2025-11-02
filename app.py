import os
import re
import asyncio
import uvicorn
import json
import httpx
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

# Markers
START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)

# =========================
# Base models
# =========================
class TranscriptSegment(BaseModel):
    text: str
    speaker: str
    start: float
    end: float

# Real-time payload models
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
# App lifespan (HTTP Client)
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

DEEPSEEK_SYSTEM_PROMPT = """You are an expert social-conversation analyst.
Your job is to read the transcript and *always* return a complete, valid, STRUCTURED JSON assessment, *no matter how short or incomplete the transcript is*.

Even if the transcript is very short (e.g., only 1-2 lines) or lacks substance, you must *still* perform your full analysis to the best of your ability:
- Do *not* use default scores. Analyze what little information you have and provide your best-effort scores for Reciprocity, Attentiveness, Warmth, Comfort, Boundary, and Chemistry.
- Do your best to generate 1-2 highlights and 1-2 improvement tips, even if they have to be more general due to the lack of context.
- Do your best to generate 1-2 `suggested_prompts`.
- Set the `confidence_score` (0-100) to reflect how confident you are in your analysis, based on the transcript's length, turn-taking, and substance. A very short transcript should result in a very low confidence score.

Return ONLY the JSON. Do not add any extra commentary or refusal text."""

def deepseek_user_prompt(transcript_json: List[Dict[str, Any]], title_hint: Optional[str]) -> str:
    schema = {
        "title": "string (<= 80 chars; use first meaningful utterance if not provided)",
        "confidence_score": "int 0-100 (See rubric for definition)",
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
        "confidence_score": "A 0-100 score of your confidence in this analysis, based on the transcript's length and substance. 0-30 = Low confidence (very short/incomplete). 30-70 = Mid confidence (partial talk). 70-100 = High confidence (rich conversation).",
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


async def call_deepseek(messages, temperature=0.2, max_tokens=1500) -> Optional[dict]:
    if not DEEPSEEK_API_KEY:
        print("❌ DeepSeek: DEEPSEEK_API_KEY is not set.")
        return None
    if not http_client:
        print("❌ DeepSeek: http_client is not initialized.")
        return None

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
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

        # --- START ROBUST PARSING FIX ---
        try:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")

            # Check if content is None or an empty string
            if not content:
                reasoning = message.get("reasoning_content", "No reasoning provided.")
                finish_reason = choice.get("finish_reason", "No finish reason.")
                print(f"❌ DeepSeek: Returned empty content. FinishReason: {finish_reason}. Reasoning: {reasoning[:500]}")
                return None

            # Content exists, now try to parse it
            return json.loads(content)
        
        except json.JSONDecodeError as e:
            # Content was not empty, but it wasn't valid JSON
            print(f"❌ Parse error (JSONDecodeError): {e}. Content preview: {content[:400]}")
            return None
        except Exception as e:
            # Other errors (e.g., KeyError if response structure is wrong)
            print(f"❌ Parse error (generic): {e}. Full response preview: {resp.text[:400]}")
            return None
        # --- END ROBUST PARSING FIX ---

    print("❌ DeepSeek: retries exhausted.")
    return None # Explicitly return None after retries


@app.get("/deepseek/ping")
async def deepseek_ping():
    msgs = [
        {"role":"system","content":"Return ONLY {\"ok\":true,\"ts\":\"<utc>\"}."},
        {"role":"user","content":"Respond with ok=true and current UTC timestamp."}
    ]
    out = await call_deepseek(msgs, temperature=0.0, max_tokens=64)
    return {"bridge_ok": bool(out), "raw": out}

# =========================
# Push helpers
# =========================
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
        resp = await http_client.post(url, headers=headers, params=params, data="")
        
        if 200 <= resp.status_code < 300:
            print("✅ Notification sent.")
            print(f"Response: {resp.text}")
        else:
            print(f"❌ Failed notification. Status: {resp.status_code}")
            print(f"Response body: {resp.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def compose_notification_from_llm(llm_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Organizes the raw LLM response into a systematic report for notifications and memory.
    Returns a (title, body) tuple.
    """
    
    def truncate(s: str, n: int = 60) -> str:
        return (s[:n].rstrip() + "…") if len(s) > n else s

    # --- Extract data with defaults ---
    title = truncate(llm_data.get("title", "Conversation Report"))
    final_score = llm_data.get("final_score", "N/A")
    confidence = llm_data.get("confidence_score", "N/A")
    
    highlights = llm_data.get("highlights", ["No highlights provided."])
    improvements = llm_data.get("improvements", ["No improvements provided."])
    prompts = llm_data.get("suggested_prompts", ["No prompts provided."])
    subscores = llm_data.get("subscores", {})

    # --- Build Body String ---
    body_lines = []
    body_lines.append(f"Date Rizz: {final_score}/100 — “{title}”")
    body_lines.append(f"AI Confidence: {confidence}/100")
    
    # --- Subscores ---
    sub_list = []
    if "reciprocity" in subscores:
        sub_list.append(f"R {subscores['reciprocity'].get('score', 'N/A')}")
    if "attentiveness" in subscores:
        sub_list.append(f"A {subscores['attentiveness'].get('score', 'N/A')}")
    if "warmth" in subscores:
        sub_list.append(f"W {subscores['warmth'].get('score', 'N/A')}")
    if "comfort" in subscores:
        sub_list.append(f"C {subscores['comfort'].get('score', 'N/A')}")
    if "boundary" in subscores:
        sub_list.append(f"B {subscores['boundary'].get('score', 'N/A')}")
    if "chemistry" in subscores:
        sub_list.append(f"Ch {subscores['chemistry'].get('score', 'N/A')}")
    
    if sub_list:
        body_lines.append(" · ".join(sub_list))

    # --- Highlights ---
    body_lines.append("\n✅ Highlights:")
    for h in highlights:
        body_lines.append(f"• {h}")

    # --- Improvements ---
    body_lines.append("\n💡 Try:")
    for i in improvements:
        body_lines.append(f"• {i}")

    # --- Prompts ---
    body_lines.append("\n💬 Next time, try:")
    for p in prompts:
        body_lines.append(f"• {p}")
        
    # --- Join and truncate ---
    final_body = "\n".join(body_lines)
    
    # Return the user-facing title and the full body
    report_title = f"Your Rizz Report: {final_score}/100 for “{title}”"
    
    if len(final_body) > 500:
        # Fallback for very long reports to fit push notification limits
        return (report_title, final_body[:497] + "…")
    
    return (report_title, final_body)

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
# Finalization (LLM-Only)
# =========================
async def llm_analyze(segments: List[TranscriptSegment], title_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    if not segments:
        return None
    transcript_json = prepare_transcript_for_llm(segments)
    messages = [
        {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
        {"role": "user", "content": deepseek_user_prompt(transcript_json, title_hint)}
    ]
    return await call_deepseek(messages)

async def _finalize_and_analyze(uid: str) -> Dict:
    state = await _get_state(uid)
    async with state.lock:
        return await _finalize_and_analyze_UNLOCKED(state, uid)

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
        # --- LLM-Only Analysis ---
        llm = await llm_analyze(clean_segments, title)
        
        if llm:
            # Generate notification body *directly* from LLM response
            report_title, report_body = compose_notification_from_llm(llm)
            
            push_uid = state.last_uid or uid
            if push_uid:
                score = llm.get('final_score', 'N/A')
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 📣 Pushing (LLM-Only) to uid='{push_uid}', score={score}")
                
                # Use the generated title and body for notification and memory
                await send_notification(push_uid, title=report_title, body=report_body)
                await save_text_as_memory(push_uid, report_body)

            # The summary returned by the API is now just the LLM JSON
            summary = {
                "status": "success",
                "summary": llm  # Return the whole LLM blob
            }
        else:
            # This 'else' now means the LLM call *failed* (network, parse error, etc.)
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⚠️ LLM unavailable. No analysis performed.")
            summary = {"status": "error", "message": "LLM analysis failed and no fallback is configured."}

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
            await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        if force_end:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🛑 force_end received for uid={effective_uid}")
            return await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        if has_start or force_start:
            if state.active and state.buffer:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔄 start while active (uid={effective_uid}) — auto-finalizing previous.")
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
            return await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        if audio_gap_trigger and state.buffer:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⏰ Large audio gap detected (> {MAX_SEG_GAP_SEC}s). Auto-finalizing (uid={effective_uid}).")
            return await _finalize_and_analyze_UNLOCKED(state, effective_uid)

        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🔄 Buffering: total segments={len(state.buffer)} (uid={effective_uid})")
        return {"status": "buffering", "segments_buffered": len(state.buffer), "will_push_on_end": True, "pid": PID}

@app.post("/conversation/end")
async def conversation_end(uid: str = Query(...)):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🛑 /conversation/end called for uid={uid}")
    return await _finalize_and_analyze(uid)

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Rizz Meter server on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
