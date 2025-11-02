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

# --- NEW METICULOUS PROMPT WITH FULL CONTEXT ---
DEEPSEEK_SYSTEM_PROMPT = """You are an expert social-conversation analyst, specializing in dating.
Your job is to read a transcript and return a single, formatted, plain-text "Post-Date Report".

*DO NOT* use JSON. *DO NOT* add any commentary, greetings, or text before or after the report.
Your entire response *must* be the report itself, starting with "Date Rizz:".

---
### YOUR GOAL & CONTEXT
You will analyze the transcript for "Date Rizz" based on key dating-relevant signals. Your analysis *must* be rooted in these signals:

1.  **Reciprocity & Turn-Taking**:
    * Is talk time balanced?
    * Is the interruption rate low?
    * Are back-channels (e.g., "mm-hmm," "yeah") timely and supportive?

2.  **Attentiveness**:
    * Is there a good question rate (not too high, not zero)?
    * Are there follow-up questions that reference prior details?
    * Does the speaker show memory of the other's preferences?

3.  **Warmth**:
    * Is the sentiment trajectory positive? Does the conversation get warmer?
    * Are there positive acknowledgments (e.g., "That's great," "Awesome")?
    * Are there appreciation phrases (e.g., "Thanks for sharing")?

4.  **Comfort & Pacing**:
    * Is the speaking rate stable and natural, not rushed or hesitant?
    * Is there good pause tolerance? Are silences comfortable or awkward?
    * Are there markers of lightness, like laughter?

5.  **Boundary Respect**:
    * Does the speaker honor topics the other person avoids?
    * Do they avoid pushing after a "no" or a clear signal of discomfort?

---
### HOW TO COMPUTE THE SCORE
1.  Analyze the transcript for all 5 signals. Normalize them per speaker and by the conversation's length.
2.  Map your findings into 0-100 sub-scores for the 5 key signals.
3.  Aggregate these into a final 0-100 "Date Rizz" score.
4.  Also generate a 0-100 "AI Confidence" score based on how much data you had (a short transcript = low confidence).

---
### REQUIRED OUTPUT FORMAT (POST-DATE MODE)
You *must* return your full analysis in this *exact* plain-text format:

Date Rizz: [Your 0-100 aggregate score] — “[A short, catchy title for the conversation]”
AI Confidence: [Your 0-100 confidence score]

[Your 1-2 sentence human summary/overview of the conversation's dynamics]

✅ Highlights:
• [Your first analysis highlight]
• [Your second analysis highlight]

💡 Try:
• [Your first improvement tip]
• [Your second improvement tip]

💬 Next time, try:
• [Your first suggested prompt for a next date]
• [Your second suggested prompt for a next date]

---
Breakdown:
• Reciprocity: [Score 0-100] ([Brief analysis, e.g., "Good balance" or "You dominated"])
• Attentiveness: [Score 0-100] ([Brief analysis, e.g., "Many good questions"])
• Warmth: [Score 0-100] ([Brief analysis, e.g., "Very positive tone"])
• Comfort: [Score 0-100] ([Brief analysis, e.g., "Easy pacing, no awkwardness"])
• Boundary: [Score 0-100] ([Brief analysis, e.g., "Respectful topic handling"])
---
✨ Highlights Reel:
• [Timestamp_s]: "[Quote from transcript]" - (Why this moment was strong)
• [Timestamp_s]: "[Another quote]" - (Why this moment was strong)

---
*IMPORTANT: Even if the transcript is very short, you MUST do your best to generate this full report. State a low confidence score if the data is poor, but *always* provide the report in this exact format.*
"""

def deepseek_user_prompt(transcript_json: List[Dict[str, Any]], title_hint: Optional[str]) -> str:
    """Creates the simple user-prompt string with the transcript."""
    transcript_str = json.dumps(transcript_json, indent=2)
    return f"Title hint (you can ignore if you find a better one): {title_hint}\n\nTranscript:\n{transcript_str}"

async def call_deepseek(messages, temperature=0.2, max_tokens=2048) -> Optional[str]: # Returns Optional[str]
    """
    Calls the DeepSeek API and returns the raw text content string,
    or None if it fails or returns empty content.
    Increased max_tokens for the larger prompt.
    """
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
        "max_tokens": max_tokens # Increased for the larger response
    }
    
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    for attempt in range(3):
        try:
            # Increased timeout for a potentially longer generation
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

        # --- MODIFIED: Return raw string, not JSON ---
        try:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") # This is now the raw string we want

            # Check if content is None or an empty string (the original problem)
            if not content:
                reasoning = message.get("reasoning_content", "No reasoning provided.")
                finish_reason = choice.get("finish_reason", "No finish reason.")
                print(f"❌ DeepSeek: Returned empty content. FinishReason: {finish_reason}. Reasoning: {reasoning[:500]}")
                return None

            # Return the raw, formatted string content
            return content.strip()
        
        except Exception as e:
            # Catch JSONDecodeError if resp.json() fails, or KeyError if structure is wrong
            print(f"❌ Parse error (generic): {e}. Full response preview: {resp.text[:400]}")
            return None
        # --- END MODIFICATION ---

    print("❌ DeepSeek: retries exhausted.")
    return None

@app.get("/deepseek/ping")
async def deepseek_ping():
    msgs = [
        {"role":"system","content":"Return ONLY {\"ok\":true,\"ts\":\"<utc>\"}."},
        {"role":"user","content":"Respond with ok=true and current UTC timestamp."}
    ]
    out_str = await call_deepseek(msgs, temperature=0.0, max_tokens=64)
    return {"bridge_ok": bool(out_str and "ok" in out_str), "raw": out_str}

# =========================
# Push helpers
# =========================
async def send_notification(uid: str, title: str, body: str):
    """
    Sends a push notification. The 'title' is a generic header,
    and 'body' is the full, formatted report from the LLM.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting push to uid={uid}")
    if not OMI_APP_ID or not OMI_APP_SECRET:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_APP_SECRET is not set.")
        return
    if not http_client:
        print("❌ HTTP Client not initialized; skipping notification.")
        return
        
    full_message = f"{title}: {body}"
    
    if len(full_message) > 500:
        full_message = full_message[:497] + "…"

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

async def create_conversation(uid: str, text: str):
    api_key = os.environ.get("OMI_API_KEY")
    if not OMI_APP_ID or not api_key or not http_client:
        return

    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/conversations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    params = {"uid": uid}
    payload = {
        "text": text,
        "text_source": "other_text",
        "text_source_spec": "rizz_meter_report",
        "language": "en"
        # optionally: started_at / finished_at in ISO 8601
    }
    resp = await http_client.post(url, headers=headers, params=params, json=payload, timeout=30.0)
    if 200 <= resp.status_code < 300:
        print("✅ Conversation created.")
    else:
        print(f"❌ Failed to create conversation. {resp.status_code} {resp.text}")



async def save_text_as_memory(uid: str, text_content: str):
    """
    Save the LLM report as a single explicit memory in Omi.
    Uses the Import API (Create Memories) per docs.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting to save memory for uid={uid}")

    api_key = os.environ.get("OMI_APP_SECRET")
    if not OMI_APP_ID or not api_key:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_API_KEY is not set. Cannot save memory.")
        return
    if not http_client:
        print("❌ HTTP Client not initialized; skipping memory save.")
        return

    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/memories"
    headers = {
        "Authorization": f"Bearer {api_key}",  # <-- API KEY (sk_...), per docs
        "Content-Type": "application/json",
    }
    params = {"uid": uid}

    # Option A: store the whole report as ONE explicit memory (recommended for your use case)
    payload = {
        "memories": [
            {
                "content": text_content,
                "tags": ["rizz_report", "dating", "post-date"]
            }
        ],
        "text_source": "other",
        "text_source_spec": "rizz_meter"
    }

    # Option B (alternative): let Omi extract memories from free text instead
    # payload = {
    #     "text": text_content,
    #     "text_source": "other",
    #     "text_source_spec": "rizz_meter"
    # }

    try:
        resp = await http_client.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=30.0
        )
        if 200 <= resp.status_code < 300:
            print("✅ Memory created (Import API).")
        else:
            print(f"❌ Failed to create memory. Status: {resp.status_code}")
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
    if seg.is_user is True: return "You"
    if seg.is_user is False: return "Partner"
    if seg.speaker is not None: return seg.speaker
    if seg.speaker_id is not None: return f"SPEAKER_{seg.speaker_id}"
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
# Finalization (LLM-Only, Direct-to-String)
# =========================
async def llm_analyze(segments: List[TranscriptSegment], title_hint: Optional[str]) -> Optional[str]:
    """
    Calls the LLM and returns the raw formatted string report.
    """
    if not segments:
        return None
    transcript_json = prepare_transcript_for_llm(segments)
    messages = [
        {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
        {"role": "user", "content": deepseek_user_prompt(transcript_json, title_hint)}
    ]
    # call_deepseek now returns a string
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
        llm_report_string = await llm_analyze(clean_segments, title) # This is now a string
        
        if llm_report_string:
            # The LLM's full, formatted report is the BODY
            report_title = "Your Rizz Report is Ready" # Generic title
            report_body = llm_report_string           # Full LLM output
            
            push_uid = state.last_uid or uid
            if push_uid:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 📣 Pushing (LLM-String) to uid='{push_uid}'")
                
                # Send notification (title + body)
                await send_notification(push_uid, title=report_title, body=report_body)
                # Save memory (just the body)
                await create_conversation(push_uid, report_body)
                await save_text_as_memory(push_uid, report_body)

                

            summary = {
                "status": "success",
                "summary": {"report": report_body} # Return the report string
            }
        else:
            # This 'else' now means the LLM call *failed* (network, parse error, empty)
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⚠️ LLM unavailable. No analysis performed.")
            summary = {"status": "error", "message": "LLM analysis failed."}

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



