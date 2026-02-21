import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import vector_store
import speech_service
import generation_service
import scoring
import scoring
from database import get_db, SocialCreds, encrypt_secret, download_db, upload_db
import social_publisher
import dateparser
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize a FastAPI application
app = FastAPI(
    title="Voice-To-Post Backend API",
    description="Foundational backend for Voice-To-Post AI-driven social media generator"
)

# Initialize the Background Scheduler
scheduler = BackgroundScheduler()

def publish_to_social_media(platform: str, text: str):
    print(f"[Scheduled Job] Publishing to {platform}: {text}")

# Configure CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GeneratePostRequest(BaseModel):
    dummy_text: str

@app.on_event("startup")
async def startup_event():
    """
    Optional: Pre-load some sample data into the index on startup
    so that searches aren't completely empty before adding new data.
    """
    # Attempt to download the latest credentials.db from Hugging Face Space Cloud Storage
    download_db()
    
    # Start the Background Scheduler
    scheduler.start()
    
    sample_data = [
        "Welcome to Voice-To-Post backend!",
        "Vector databases help in doing semantic similarity search.",
        "FastAPI is a fast, highly performant web framework for building APIs."
    ]
    vector_store.add_text_to_index(sample_data)
    print("Application initialized. Loaded sample data into the vector store.")

@app.get("/")
async def health_endpoint():
    """
    Health Endpoint confirming the backend is running.
    """
    return {"status": "Voice-To-Post Backend is running"}

class SaveKeysRequest(BaseModel):
    platform: str
    client_id: str
    client_secret: str

@app.post("/settings/save-keys")
async def save_keys(request: SaveKeysRequest, db: Session = Depends(get_db)):
    """
    Saves or updates OAuth2 client credentials for a specific social platform.
    """
    platform_key = request.platform.lower()
    
    # Check if creds for this platform already exist
    existing_creds = db.query(SocialCreds).filter(SocialCreds.platform == platform_key).first()
    
    if existing_creds:
        existing_creds.client_id = request.client_id
        existing_creds.encrypted_secret = encrypt_secret(request.client_secret)
        db.commit()
        # Upload sync to permanent HF Dataset
        upload_db()
        return {"status": "success", "message": f"Updated credentials for {platform_key}."}
    else:
        new_creds = SocialCreds(
            platform=platform_key,
            client_id=request.client_id,
            encrypted_secret=encrypt_secret(request.client_secret)
        )
        db.add(new_creds)
        db.commit()
        # Upload sync to permanent HF Dataset
        upload_db()
        return {"status": "success", "message": f"Saved new credentials for {platform_key}."}

class ParseScheduleRequest(BaseModel):
    transcript: str

@app.post("/parse-schedule")
async def parse_schedule(audio_file: UploadFile = File(...)):
    """Transcribes schedule audio and returns ISO 8601 time"""
    # 1. Transcribe the audio
    audio_bytes = await audio_file.read()
    transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    
    # 2. Parse the human text into datetime
    parsed_time = dateparser.parse(
        transcript, 
        settings={'TIMEZONE': 'Asia/Kolkata', 'RETURN_AS_TIMEZONE_AWARE': True}
    )
    if not parsed_time:
        raise HTTPException(status_code=400, detail="Could not parse scheduled time from transcript.")
    
    return {
        "parsed_time": parsed_time.isoformat(),
        "human_text": transcript
    }

class ConfirmPostRequest(BaseModel):
    platform: str
    text: str
    scheduled_time: Optional[str] = None

@app.post("/confirm-post")
async def confirm_post(request: ConfirmPostRequest):
    if request.scheduled_time:
        try:
            dt = datetime.fromisoformat(request.scheduled_time)
            scheduler.add_job(
                publish_to_social_media, 
                'date', 
                run_date=dt, 
                args=[request.platform, request.text]
            )
            return {"status": "scheduled", "message": f"Post scheduled for {dt.isoformat()}"}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid scheduled_time format. Must be ISO 8601.")
    else:
        publish_to_social_media(request.platform, request.text)
        return {"status": "published_immediately", "message": "Post published immediately."}

@app.post("/generate-post")
async def generate_post(
    audio_file: UploadFile = File(...),
    tone: str = Form(...) # Added Tone Parameter
):
    """Transcribes audio, applies tone, checks safety, and returns variations"""
    audio_bytes = await audio_file.read()
    transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    
    if transcript.startswith("Error") or transcript.startswith("ERROR"):
        raise HTTPException(status_code=500, detail=transcript)
        
    results = vector_store.search_index(transcript, top_k=3)
    avg_distance = sum([res["distance"] for res in results]) / len(results) if results else -1.0
    
    # Pass the 'tone' to your Gemini generation service to get the 5 variations
    # (Ensure generation_service is updated to return a list of 5 dictionaries)
    generated_variations = await generation_service.generate_post_rag(transcript, results, tone=tone)
    
    # Simplified Gatekeeper Check
    score_data = scoring.calculate_safety_score(generated_variations[0]['text'], avg_distance)
    c_score = score_data["final_score"]
    
    if c_score >= 0.75:
        # DO NOT PUBLISH HERE. Just return the variations for the UI Carousel!
        return {
            "status": "success",
            "variations": generated_variations, 
            "error": None
        }
    else:
        return {
            "status": "rejected",
            "variations": None,
            "error": "The generated post failed safety thresholds (Score < 0.75)."
        }
