import uvicorn
from fastapi import FastAPI, Request
import json
from pydantic import BaseModel, Field
from typing import List, Optional
import requests  # <-- 1. IMPORT THE NEW LIBRARY
import os

# --- Data Models (Unchanged) ---
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
    id: str  # We keep this as 'str' from our last fix
    started_at: str
    finished_at: str
    transcript_segments: List[TranscriptSegment]
    structured: StructuredMemory
    apps_response: Optional[List[dict]] = Field(alias="apps_response", default=[])


# --- Initialize App (Unchanged) ---
app = FastAPI(title="Rizz Meter Server")


# --- Analysis Functions (Unchanged) ---
# (All your functions like analyze_reciprocity, analyze_attentiveness, etc.
#  go here. I'm omitting them for brevity, but you should
#  paste them in from your last file.)
def analyze_reciprocity(segments: List[TranscriptSegment]):
    # ... (paste your code here)
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
    # ... (paste your code here)
    questions = {}
    for seg in segments:
        if seg.speaker not in questions: questions[seg.speaker] = 0
        if seg.text.strip().endswith("?"): questions[seg.speaker] += 1
    total_questions = sum(questions.values())
    score = min(total_questions * 10, 100)
    return {"score": score, "total_questions": total_questions, "breakdown": questions}

def analyze_warmth(segments: List[TranscriptSegment]):
    # ... (paste your code here)
    POSITIVE_WORDS = ["great", "awesome", "cool", "love", "amazing", "thank you", "thanks", "wonderful", "fantastic"]
    warmth_count = 0
    for seg in segments:
        text_lower = seg.text.lower()
        for word in POSITIVE_WORDS:
            if word in text_lower: warmth_count += 1
    score = min(warmth_count * 8, 100)
    return {"score": score, "positive_word_count": warmth_count}

def analyze_comfort(segments: List[TranscriptSegment]):
    # ... (paste your code here)
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
    

# --- 3. NEW NOTIFICATION FUNCTION ---

# !! IMPORTANT !!
# You get this key from the Omi developer portal when you
# register your app. This is just a placeholder.
OMI_API_KEY = OMI_API_KEY = os.environ.get("OMI_API_KEY")
NOTIFICATION_URL = "https://api.omi.me/v1/notifications"

def send_notification(uid: str, title: str, body: str):
    """
    Sends a push notification to a specific user via the Omi API.
    """
    print(f"Attempting to send notification to user: {uid}")
    
    headers = {
        "Authorization": f"Bearer {OMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "uid": uid,
        "title": title,
        "body": body
    }
    
    try:
        response = requests.post(NOTIFICATION_URL, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("✅ Notification sent successfully!")
        else:
            print(f"❌ Failed to send notification. Status: {response.status_code}")
            print(f"Response body: {response.text}")
            
    except Exception as e:
        print(f"Error sending notification: {e}")


# --- 4. UPDATED WEBHOOK ENDPOINT ---

@app.post("/rizz-meter")
async def analyze_memory(memory: Memory, uid: str): # <-- 2. 'uid: str' IS NEW
    """
    This function will be called every time Omi sends
    a new memory. It now ALSO captures the 'uid' from
    the URL (e.g., /rizz-meter?uid=user-123)
    """
    print(f"🎉 Analyzing Memory: {memory.structured.title} for user: {uid}")
    
    segments = memory.transcript_segments
    
    if len(segments) < 2:
        print("Not enough segments to analyze.")
        return {"status": "error", "message": "Not enough segments."}

    # Run all our analyses (unchanged)
    reciprocity = analyze_reciprocity(segments)
    attentiveness = analyze_attentiveness(segments)
    warmth = analyze_warmth(segments)
    comfort = analyze_comfort(segments)
    
    # Calculate final score (unchanged)
    final_score = (
        reciprocity.get('score', 0) +
        attentiveness.get('score', 0) +
        warmth.get('score', 0) +
        comfort.get('score', 0)
    ) / 4
    
    # --- 5. CALL THE NOTIFICATION FUNCTION ---
    # Instead of just printing, we now send the notification.
    
    report_title = "Your Rizz Report is Ready!"
    report_body = f"Your final score for '{memory.structured.title}' is {round(final_score)}/100."
    
    send_notification(uid, title=report_title, body=report_body)
    
    # We'll return a simple success message
    return {"status": "success", "message_sent": report_body}


# --- Run Server (Unchanged) ---
if __name__ == "__main__":
    print("Starting Rizz Meter server on http://localhost:8000")

    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)

