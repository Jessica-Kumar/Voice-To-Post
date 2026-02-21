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

# Initialize a FastAPI application
app = FastAPI(
    title="Voice-To-Post Backend API",
    description="Foundational backend for Voice-To-Post AI-driven social media generator"
)

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

@app.post("/generate-post")
async def generate_post(
    platform: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Generates a social media post from an audio file upload via Deepgram STT, 
    FAISS RAG context retrieval, and Gemini LLM. Validates using Safety Gatekeeper.
    """
    # 1. Transcribe the uploaded audio with Deepgram
    audio_bytes = await audio_file.read()
    transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    
    if transcript.startswith("Error") or transcript.startswith("ERROR"):
        raise HTTPException(status_code=500, detail=transcript)
        
    # 2. Retrieve Context via FAISS RAG
    # We use the transcript as the query to find similar past thoughts/posts
    results = vector_store.search_index(transcript, top_k=3)
    
    # Calculate an average "context distance" for scoring later
    avg_distance = sum([res["distance"] for res in results]) / len(results) if results else -1.0
    
    # 3. Generate Post via Gemini (LangChain)
    generated_post = await generation_service.generate_post_rag(transcript, results)
    
    if generated_post.startswith("Error") or generated_post.startswith("ERROR"):
        raise HTTPException(status_code=500, detail=generated_post)
        
    # 4. Evaluate Post using Safety Gatekeeper (Threshold T=0.75)
    score_data = scoring.calculate_safety_score(generated_post, avg_distance)
    c_score = score_data["final_score"]
    
    if c_score >= 0.75:
        # Passed gatekeeper, attempt to publish
        publish_result = await social_publisher.publish_to_platform(platform, generated_post)
        
        return {
            "status": "success",
            "transcript": transcript,
            "generated_post": generated_post,
            "gatekeeper_score": score_data,
            "publish_result": publish_result
        }
    else:
        # Failed gatekeeper
        return {
            "status": "rejected",
            "message": "The generated post did not meet the safety and quality thresholds (Score < 0.75).",
            "gatekeeper_score": score_data
        }
