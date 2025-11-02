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
DEEPSEEK_NUDGE_MODEL = os.environ.get("DEEPSEEK_NUDGE_MODEL", "deepseek-chat")

# Auto-finalization knobs
IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "180"))
MAX_SEG_GAP_SEC = float(os.environ.get("MAX_SEG_GAP_SEC", "120"))
PID = os.getpid()

# Silence/Nudge knobs
SILENCE_HINT_SEC   = int(os.environ.get("SILENCE_HINT_SEC", "20"))     # when to nudge
BUZZ_COOLDOWN_SEC  = int(os.environ.get("BUZZ_COOLDOWN_SEC", "120"))   # avoid spam
NUDGE_MAX_CHARS    = int(os.environ.get("NUDGE_MAX_CHARS", "160"))     # short push text

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

# --- MAIN REPORT PROMPT ---
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

# --- NUDGE & PROFILE PROMPTS ---
DEEPSEEK_NUDGE_SYSTEM_PROMPT = """You are a concise conversation coach.
ALWAYS produce exactly ONE line (<= {max_chars} chars) that begins with 'Try:' and suggests 2–3 adjacent topics.
• Do this REGARDLESS of transcript length or who is speaking.
• Prioritize the other person (not the user): frame topics as questions or prompts about THEM.
• If the partner's details are unknown, assume a neutral 'your date' persona.
• Use ' • ' to separate topics. No extra sentences, no emojis, no newlines."""
DEEPSEEK_PROFILE_SYSTEM_PROMPT = """Extract a partner profile from a date transcript.
Return ONLY minified JSON with these keys:
{
  "name": null|string,
  "birthday": null|string,
  "hobbies": string[],
  "likes": string[],
  "dislikes": string[],
  "work": null|string,
  "location": null|string,
  "values": string[],
  "green_flags": string[],
  "red_flags": string[],
  "other_facts": string[]
}
If unknown, use null or []. No commentary. Only JSON."""

def deepseek_user_prompt(transcript_json: List[Dict[str, Any]], title_hint: Optional[str]) -> str:
    """Creates the simple user-prompt string with the transcript."""
    transcript_str = json.dumps(transcript_json, indent=2)
    return f"Title hint (you can ignore if you find a better one): {title_hint}\n\nTranscript:\n{transcript_str}"

async def call_deepseek(messages, temperature=0.2, max_tokens=2048, *, model: Optional[str] = None, stop: Optional[List[str]] = None) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        print("❌ DeepSeek: DEEPSEEK_API_KEY is not set.")
        return None
    if not http_client:
        print("❌ DeepSeek: http_client is not initialized.")
        return None

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": (model or DEEPSEEK_MODEL),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if stop:
        payload["stop"] = stop

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
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            if not content:
                reasoning = message.get("reasoning_content", "No reasoning provided.")
                finish_reason = choice.get("finish_reason", "No finish reason.")
                print(f"❌ DeepSeek: Returned empty content. FinishReason: {finish_reason}. Reasoning: {reasoning[:500]}")
                return None
            return content.strip()
        except Exception as e:
            print(f"❌ Parse error (generic): {e}. Full response preview: {resp.text[:400]}")
            return None

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
# Token helper (Imports)
# =========================
def _get_imports_token() -> Optional[str]:
    """
    Imports (create conversation/memories) expect an API key (sk_...).
    If you haven't split creds yet, we fall back to OMI_APP_SECRET.
    """
    return os.environ.get("OMI_API_KEY") or os.environ.get("OMI_APP_SECRET")

# =========================
# Push helpers
# =========================
async def create_conversation(uid: str, text: str):
    api_key = _get_imports_token()
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
    }
    resp = await http_client.post(url, headers=headers, params=params, json=payload, timeout=30.0)
    if 200 <= resp.status_code < 300:
        print("✅ Conversation created.")
    else:
        print(f"❌ Failed to create conversation. {resp.status_code} {resp.text}")

# Notifications: keep using App Secret (per docs)
async def send_notification(uid: str, title: str, body: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting push to uid={uid}")
    if not OMI_APP_ID or not OMI_APP_SECRET or not http_client:
        print("❌ Missing OMI creds or HTTP client.")
        return

    full_message = f"{title}: {body}"

    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    headers = {
        "Authorization": f"Bearer {OMI_APP_SECRET}",
        "Content-Type": "application/json",
        "Content-Length": "0",
    }
    params = {"uid": uid, "message": full_message}

    try:
        resp = await http_client.post(url, headers=headers, params=params, timeout=15.0)
        if 200 <= resp.status_code < 300:
            print("✅ Notification sent.", resp.text)
        else:
            print(f"❌ Failed notification. {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")

# Imports: save LLM report as conversation + explicit memory
async def save_text_as_memory(uid: str, text_content: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) Attempting to save memory for uid={uid}")

    api_key = _get_imports_token()
    if not OMI_APP_ID or not api_key or not http_client:
        print("❌ Missing OMI_APP_ID or Imports token or HTTP client.")
        return

    # (A) Create a conversation holding the full report text
    try:
        conv_url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/conversations"
        conv_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        conv_params = {"uid": uid}
        conv_payload = {
            "text": text_content,
            "text_source": "other_text",
            "text_source_spec": "rizz_meter_report",
            "language": "en",
        }
        conv_resp = await http_client.post(conv_url, headers=conv_headers, params=conv_params,
                                           json=conv_payload, timeout=30.0)
        if 200 <= conv_resp.status_code < 300:
            print("✅ Conversation created (Imports).")
        else:
            print(f"❌ Create conversation failed. {conv_resp.status_code} {conv_resp.text}")
    except Exception as e:
        print(f"Error creating conversation: {e}")

    # (B) Create an explicit memory: short, searchable summary + tags
    try:
        mem_url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/memories"
        mem_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        mem_params = {"uid": uid}

        first_line = (text_content.splitlines() or [""])[0]
        memory_content = first_line[:300]  # keep memory concise

        mem_payload = {
            "text": text_content,
            "text_source": "other",
            "text_source_spec": "rizz_meter",
            "memories": [
                {
                    "content": memory_content,
                    "tags": ["rizz_report", "dating", "post-date"]
                }
            ]
        }

        mem_resp = await http_client.post(mem_url, headers=mem_headers, params=mem_params,
                                          json=mem_payload, timeout=30.0)
        if 200 <= mem_resp.status_code < 300:
            print("✅ Memory created (Imports).")
        else:
            print(f"❌ Failed to create memory. Status: {mem_resp.status_code}")
            print(f"Response body: {mem_resp.text}")
    except Exception as e:
        print(f"Error saving memory: {e}")

# --- NEW: save partner profile memory ---
async def save_partner_profile_memory(uid: str, profile: dict):
    api_key = _get_imports_token()
    if not OMI_APP_ID or not api_key or not http_client or not profile:
        return

    def _join(items):
        return ", ".join([x for x in items if isinstance(x, str) and x.strip()]) if items else ""

    summary = []
    if profile.get("birthday"): summary.append(f"Birthday: {profile['birthday']}")
    if _join(profile.get("hobbies")): summary.append(f"Hobbies: {_join(profile.get('hobbies'))}")
    if _join(profile.get("likes")): summary.append(f"Likes: {_join(profile.get('likes'))}")
    if _join(profile.get("dislikes")): summary.append(f"Dislikes: {_join(profile.get('dislikes'))}")
    if profile.get("work"): summary.append(f"Work: {profile['work']}")
    if profile.get("location"): summary.append(f"Location: {profile['location']}")
    if _join(profile.get("values")): summary.append(f"Values: {_join(profile.get('values'))}")
    content_line = "Partner profile — " + " | ".join(summary) if summary else "Partner profile — (minimal data)"

    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/memories"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    params = {"uid": uid}
    payload = {
        "text": content_line,
        "text_source": "other",
        "text_source_spec": "rizz_meter_partner_profile",
        "memories": [
            {
                "content": content_line,
                "tags": ["partner_profile", "dating", "rizz_meter"]
            }
        ]
    }
    resp = await http_client.post(url, headers=headers, params=params, json=payload, timeout=30.0)
    if 200 <= resp.status_code < 300:
        print("✅ Partner profile memory saved.")
    else:
        print(f"❌ Partner profile memory failed. {resp.status_code} {resp.text}")

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
        # Silence-nudge
        self.silence_task: Optional[asyncio.Task] = None
        self.last_buzz_wall_ts: float = 0.0

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
# LLM analysis + nudges + profile extraction
# =========================
def _compact_transcript_text(segments: List[TranscriptSegment], max_chars: int = 1200) -> str:
    parts = [f"{s.speaker}: {s.text.strip()}" for s in segments if s.text and s.text.strip()]
    out = "\n".join(parts[-50:])
    return out[-max_chars:]

async def generate_topic_nudges(segments: List[TranscriptSegment], title_hint: Optional[str]) -> Optional[str]:
    # Build a compact body (OK even if it's only user lines or very short)
    body = _compact_transcript_text(segments or [], 1200)

    # Try to find a non-user speaker to personalize; otherwise use a neutral label
    target = next(
        (s.speaker for s in reversed(segments or [])
         if (s.speaker or "").lower() != "you" and (s.text or "").strip()),
        "your date"
    )

    sys = DEEPSEEK_NUDGE_SYSTEM_PROMPT.replace("{max_chars}", str(NUDGE_MAX_CHARS))
    user_content = (
        f"Direct your suggestions toward {target}. "
        f"If context is thin, infer plausible adjacent topics from general small-talk heuristics.\n\n"
        f"Title hint: {title_hint or 'Date'}\n\nTranscript:\n{body}"
    )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_content}
    ]

    # Use deepseek-chat (not reasoner) + stop at newline to force a single line
    text = await call_deepseek(
        messages,
        temperature=0.6,
        max_tokens=64,
        model=DEEPSEEK_NUDGE_MODEL,   # 'deepseek-chat' by default
        stop=["\n"]
    )

    # Fallbacks to guarantee an output even if API returns empty content
    if not text:
        fallback_sys = (
            f"ALWAYS respond with exactly one line starting with 'Try:' and <= {NUDGE_MAX_CHARS} chars. "
            "If details are unknown, prefer neutral but specific, partner-centered prompts."
        )
        text = await call_deepseek(
            [{"role": "system", "content": fallback_sys}, {"role": "user", "content": user_content}],
            temperature=0.5,
            max_tokens=64,
            model="deepseek-chat",
            stop=["\n"]
        )

    if not text:
        # Absolute last resort: deterministic, partner-focused default
        return "Try: their weekend plans • favorite cuisines • a recent show they liked"

    line = text.strip().splitlines()[0]
    if not line.lower().startswith("try:"):
        line = "Try: " + line
    return line[:NUDGE_MAX_CHARS]

async def extract_partner_profile(segments: List[TranscriptSegment]) -> Optional[dict]:
    if not DEEPSEEK_API_KEY or not segments:
        return None
    tx = _compact_transcript_text(segments, 4000)
    messages = [
        {"role": "system", "content": DEEPSEEK_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": tx}
    ]
    raw = await call_deepseek(messages, temperature=0.0, max_tokens=768)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        raw_clean = raw.strip().strip("`").strip()
        try:
            return json.loads(raw_clean)
        except Exception:
            print(f"⚠️ profile JSON parse failed. preview={raw[:200]}")
            return None

async def llm_analyze(segments: List[TranscriptSegment], title_hint: Optional[str]) -> Optional[str]:
    if not segments:
        return None
    transcript_json = prepare_transcript_for_llm(segments)
    messages = [
        {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
        {"role": "user", "content": deepseek_user_prompt(transcript_json, title_hint)}
    ]
    return await call_deepseek(messages)

# =========================
# Silence monitor
# =========================
async def _silence_monitor(uid: str):
    try:
        while True:
            await asyncio.sleep(3)
            state = await _get_state(uid)
            with_data: Optional[List[TranscriptSegment]] = None
            title_hint: Optional[str] = None
            now_ts = datetime.now(timezone.utc).timestamp()
            async with state.lock:
                if not state.active:
                    break
                silent_for = now_ts - (state.last_wall_ts or now_ts)
                can_buzz = (now_ts - (state.last_buzz_wall_ts or 0.0)) >= BUZZ_COOLDOWN_SEC
                if silent_for >= SILENCE_HINT_SEC and can_buzz and state.buffer:
                    with_data = list(state.buffer[-10:])
                    title_hint = state.title
                    state.last_buzz_wall_ts = now_ts

            if with_data:
                suggestion = await generate_topic_nudges(with_data, title_hint)
                if not suggestion:
                    suggestion = "Try: weekend plans • favorite food • a recent movie/show"
                await send_notification(uid, title="💡 Conversation nudge", body=suggestion)
    except asyncio.CancelledError:
        pass

def _ensure_silence_monitor(state: ConvState, uid: str):
    if (state.silence_task is None) or state.silence_task.done():
        state.silence_task = asyncio.create_task(_silence_monitor(uid))

# =========================
# Finalization (LLM-Only, Direct-to-String)
# =========================
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
        llm_report_string = await llm_analyze(clean_segments, title)

        if llm_report_string:
            report_title = "Your Rizz Report is Ready"
            report_body = llm_report_string

            push_uid = state.last_uid or uid
            if push_uid:
                print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 📣 Pushing (LLM-String) to uid='{push_uid}'")
                await send_notification(push_uid, title=report_title, body=report_body)
                await create_conversation(push_uid, report_body)
                await save_text_as_memory(push_uid, report_body)

                # Extract + save partner profile memory
                try:
                    profile = await extract_partner_profile(clean_segments)
                    if profile:
                        await save_partner_profile_memory(push_uid, profile)
                except Exception as e:
                    print(f"⚠️ profile extraction/save error: {e}")

            summary = {"status": "success", "summary": {"report": report_body}}
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) ⚠️ LLM unavailable. No analysis performed.")
            summary = {"status": "error", "message": "LLM analysis failed."}

    # Reset state
    state.active = False
    state.buffer = []
    state.title = None
    state.last_uid = None
    state.touch_wall()
    state.last_seg_end = 0.0
    # stop silence monitor
    task = state.silence_task
    state.silence_task = None
    if task and not task.done():
        task.cancel()

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
            state.last_buzz_wall_ts = 0.0
            print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟢 conversation starts — buffering begins (uid={effective_uid})")
            _ensure_silence_monitor(state, effective_uid)

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



