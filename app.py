import uvicorn
from fastapi import FastAPI, Request
import json
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
import os  # <-- Make sure 'os' is imported

# --- 1. Get Secure Keys from Environment (Updated) ---
# We now get BOTH keys from Render's environment
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")


# --- 2. Data Models (Unchanged) ---
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


# --- 3. Initialize App (Unchanged) ---
app = FastAPI(title="Rizz Meter Server")


# --- 4. Analysis Helper Functions (Unchanged) ---
# (Paste all your `analyze_...` functions here. I've re-included them.)

def analyze_reciprocity(segments: List[TranscriptSegment]):
    talk_time = {}
    for seg in segments:
        duration = seg.end - seg.start
        if seg.speaker not in talk_time: talk_time[seg.speaker] = 0.0
        talk_time[seg.speaker] += duration
    speakers = list(talk_time.keys())
    if len(speakers) != 2: return {"error": "Not a 2-person conversation", "talk_time": talk_time}
    speaker_0_time = talk_time[speakers[0]]
    speaker_1_time = talk_time[speakers[1]]
    total_time = speaker_0_time + speaker_1_time
    try: balance = speaker_0_time / total_time
    except ZeroDivisionError: balance = 0.5
    reciprocity_score = (1 - abs(balance - 0.5) * 2) * 100
    return {"score": round(reciprocity_score), "balance": f"{round(balance * 100)}% / {round((1-balance) * 100)}%"}

def analyze_attentiveness(segments: List[TranscriptSegment]):
    questions = {}
    for seg in segments:
        if seg.speaker not in questions: questions[seg.speaker] = 0
        if seg.text.strip().endswith("?"): questions[seg.speaker] += 1
    total_questions = sum(questions.values())
    score = min(total_questions * 10, 100)
    return {"score": score, "total_questions": total_questions, "breakdown": questions}

def analyze_warmth(segments: List[TranscriptSegment]):
    POSITIVE_WORDS = ["great", "awesome", "cool", "love", "amazing", "thank you", "thanks", "wonderful", "fantastic"]
    warmth_count = 0
    for seg in segments:
        text_lower = seg.text.lower()
        for word in POSITIVE_WORDS:
            if word in text_lower: warmth_count += 1
    score = min(warmth_count * 8, 100)
    return {"score": score, "positive_word_count": warmth_count}

def analyze_comfort(segments: List[TranscriptSegment]):
    pauses = []
    if len(segments) < 2: return {"score": 50, "average_pause_s": 0}
    for i in range(1, len(segments)):
        prev_seg, curr_seg = segments[i-1], segments[i]
        pause_duration = curr_seg.start - prev_seg.end
        if curr_seg.speaker != prev_seg.speaker and 0 < pause_duration < 10.0:
            pauses.append(pause_duration)
    if not pauses: return {"score": 60, "average_pause_s": 0}
    avg_pause = sum(pauses) / len(pauses)
    if avg_pause < 1.0: score = (avg_pause / 1.0) * 100
    else: score = max(0, (1 - (avg_pause - 1.0) / 2.0)) * 100
    return {"score": round(score), "average_pause_s": round(avg_pause, 2)}


# --- 5. NEW NOTIFICATION FUNCTION (The Fix) ---

def send_notification(uid: str, title: str, body: str):
    """
    Sends a direct notification to an Omi user
    using the v2 /integrations API.
    """
    print(f"Attempting to send v2 notification to user: {uid}")
    
    if not OMI_APP_ID or not OMI_APP_SECRET:
        print("❌ CRITICAL ERROR: OMI_APP_ID or OMI_APP_SECRET is not set in environment.")
        return

    # Per the docs, the message is a combination of title and body
    full_message = f"{title}: {body}"

    # Build the URL and headers as specified in the documentation
    notification_url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    
    headers = {
        "Authorization": f"Bearer {OMI_APP_SECRET}",
        "Content-Type": "application/json",
        "Content-Length": "0"  # <-- This is the strange but required part
    }
    
    # Per the docs, data is sent as query parameters
    params = {
        "uid": uid,
        "message": full_message
    }
    
    try:
        # Note: We send an empty `data` payload because Content-Length is 0
        response = requests.post(notification_url, headers=headers, params=params, data="")
        
        if 200 <= response.status_code < 300:
            print("✅ Notification sent successfully!")
            print(f"Response: {response.text}")
        else:
            print(f"❌ Failed to send notification. Status: {response.status_code}")
            print(f"Response body: {response.text}")
            
    except Exception as e:
        print(f"Error sending notification: {e}")


# --- 6. UPDATED WEBHOOK ENDPOINT (Unchanged from last time) ---
# This part is the same as our last version.
@app.post("/memory_created")
async def analyze_memory(memory: Memory, uid: str):
    print(f"🎉 Analyzing Memory: {memory.structured.title} for user: {uid}")
    
    segments = memory.transcript_segments
    
    if len(segments) < 2:
        print("Not enough segments to analyze.")
        return {"status": "error", "message": "Not enough segments."}

    reciprocity = analyze_reciprocity(segments)
    attentiveness = analyze_attentiveness(segments)
    warmth = analyze_warmth(segments)
    comfort = analyze_comfort(segments)
    
    final_score = (
        reciprocity.get('score', 0) +
        attentiveness.get('score', 0) +
        warmth.get('score', 0) +
        comfort.get('score', 0)
    ) / 4

    report_title = "Your Rizz Report is Ready!"
    report_body = f"Your final score for '{memory.structured.title}' is {round(final_score)}/100."
    
    # Call our new, correct notification function
    send_notification(uid, title=report_title, body=report_body)
    
    return {"status": "success", "message_sent": report_body}


# --- 7. Run Server (Unchanged) ---
if __name__ == "__main__":
    # Use port 10000 for Render
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Rizz Meter server on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

