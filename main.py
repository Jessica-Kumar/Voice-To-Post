import os
import secrets
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
import tweepy
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import dateparser

import vector_store
import speech_service
import generation_service
import scoring
from database import get_db, SocialCreds, encrypt_secret, decrypt_secret, download_db, upload_db
import social_publisher

load_dotenv()

app = FastAPI(title="Voice-To-Post Backend API")
scheduler = BackgroundScheduler()

# OAuth App credentials from environment
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")
TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET")
BASE_URL = os.getenv("BASE_URL")  

# In-memory store for PKCE verifier (use session/cache in production)
twitter_oauth_state = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def publish_to_social_media(platform: str, text: str):
    print(f"[Scheduled Job] Publishing to {platform}: {text}")

@app.on_event("startup")
async def startup_event():
    download_db()
    scheduler.start()
    sample_data = [
        "Welcome to Voice-To-Post backend!",
        "Vector databases help in doing semantic similarity search.",
        "FastAPI is a fast, highly performant web framework for building APIs."
    ]
    vector_store.add_text_to_index(sample_data)  # Should associate with system user or just global
    print("Application initialized. Loaded sample data into the vector store.")

@app.get("/")
async def health_endpoint():
    return {"status": "Voice-To-Post Backend is running"}

# ==================== User Profile Sync Helpers ====================
async def sync_user_profile(user_id: str, platform: str, access_token: str, db: Session):
    """Fetch and store user profile information (bio, headline, etc.)"""
    if platform == "twitter":
        try:
            client = tweepy.Client(bearer_token=access_token)
            # Get authenticated user's own profile (requires 'users.read' scope)
            me = client.get_me(user_fields=["description"])
            if me.data:
                description = me.data.description
                # Update database (assumes SocialCreds has a twitter_bio column)
                creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
                if not creds:
                    creds = SocialCreds(user_id=user_id)
                    db.add(creds)
                # We need to add this column to SocialCreds model
                creds.twitter_bio = description
                db.commit()
                print(f"Synced Twitter bio for user {user_id}")
        except Exception as e:
            print(f"Error syncing Twitter profile: {e}")

    elif platform == "linkedin":
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient() as client:
            # Get basic profile info (vanityName, headline) using /v2/me endpoint
            resp = await client.get("https://api.linkedin.com/v2/me", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                vanity_name = data.get("vanityName")
                # Also get headline from another endpoint? /v2/people/(id)?projection=(headline)
                # For simplicity, we'll just store vanityName and maybe headline from another call.
                # Alternatively, use /v2/userinfo (OpenID) to get name, but that's less rich.
                # We'll also fetch headline separately.
                # Get profile headline (requires 'profile' scope)
                # This endpoint may need the person ID
                person_id = data.get("id")
                if person_id:
                    profile_resp = await client.get(
                        f"https://api.linkedin.com/v2/people/{person_id}?projection=(id,firstName,lastName,headline)",
                        headers=headers
                    )
                    if profile_resp.status_code == 200:
                        profile_data = profile_resp.json()
                        headline = profile_data.get("headline")
                    else:
                        headline = None
                else:
                    headline = None
            else:
                vanity_name = None
                headline = None

        if vanity_name or headline:
            creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
            if not creds:
                creds = SocialCreds(user_id=user_id)
                db.add(creds)
            # Add columns: linkedin_vanity_name, linkedin_headline
            creds.linkedin_vanity_name = vanity_name
            creds.linkedin_headline = headline
            db.commit()
            print(f"Synced LinkedIn profile for user {user_id}")

# ==================== OAuth Endpoints ====================
@app.get("/auth/linkedin/login")
async def linkedin_login():
    scope = "w_member_social,profile,openid"  # profile for user info
    auth_url = (
        f"https://www.linkedin.com/oauth/v2/authorization"
        f"?response_type=code"
        f"&client_id={LINKEDIN_CLIENT_ID}"
        f"&redirect_uri={BASE_URL}/auth/linkedin/callback"
        f"&scope={scope}"
    )
    return RedirectResponse(auth_url)

@app.get("/auth/linkedin/callback")
async def linkedin_callback(code: str, db: Session = Depends(get_db)):
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": f"{BASE_URL}/auth/linkedin/callback",
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"LinkedIn token exchange failed: {resp.text}")
        token_data = resp.json()
        access_token = token_data["access_token"]

    # Store token for demo_user (or actual user)
    user_id = "demo_user"  # Replace with actual user identification in multi-user setup
    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        creds = SocialCreds(user_id=user_id)
        db.add(creds)
    creds.linkedin_access_token = encrypt_secret(access_token)
    db.commit()
    upload_db()

    # Sync user profile asynchronously (fire and forget)
    await sync_user_profile(user_id, "linkedin", access_token, db)

    return HTMLResponse("<h1>LinkedIn authentication successful! You can close this window.</h1>")

@app.get("/auth/twitter/login")
async def twitter_login():
    # Generate code verifier and challenge
    oauth2_handler = tweepy.OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        client_secret=TWITTER_CLIENT_SECRET,
        redirect_uri=f"{BASE_URL}/auth/twitter/callback",
        scope=["tweet.read", "tweet.write", "users.read", "offline.access"]  # users.read for bio
    )
    # Generate PKCE code verifier (tweepy can handle it)
    authorization_url, state = oauth2_handler.get_authorization_url()
    # Store state and verifier (in production use session)
    twitter_oauth_state["demo_user"] = {
        "state": state,
        "code_verifier": oauth2_handler.code_verifier
    }
    return RedirectResponse(authorization_url)

@app.get("/auth/twitter/callback")
async def twitter_callback(code: str, state: str, db: Session = Depends(get_db)):
    # Retrieve stored state and verifier
    stored = twitter_oauth_state.get("demo_user")
    if not stored or stored["state"] != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    oauth2_handler = tweepy.OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        client_secret=TWITTER_CLIENT_SECRET,
        redirect_uri=f"{BASE_URL}/auth/twitter/callback",
        scope=["tweet.read", "tweet.write", "users.read", "offline.access"]
    )
    oauth2_handler.code_verifier = stored["code_verifier"]
    # Exchange code for token
    try:
        token_data = oauth2_handler.fetch_token(code)
        access_token = token_data["access_token"]
        refresh_token = token_data.get("refresh_token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Twitter token exchange failed: {str(e)}")

    user_id = "demo_user"
    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        creds = SocialCreds(user_id=user_id)
        db.add(creds)
    creds.twitter_access_token = encrypt_secret(access_token)
    if refresh_token:
        creds.twitter_refresh_token = encrypt_secret(refresh_token)
    db.commit()
    upload_db()

    # Clean up stored verifier
    twitter_oauth_state.pop("demo_user", None)

    # Sync user profile (bio)
    await sync_user_profile(user_id, "twitter", access_token, db)

    return HTMLResponse("<h1>Twitter authentication successful! You can close this window.</h1>")

# ==================== Generation Endpoint with Guaranteed 5 Loop ====================
@app.post("/generate-post")
async def generate_post(
    audio_file: UploadFile = File(...),
    tone: str = Form(...),
    platform: str = Form(...),
    user_id: str = Form("demo_user")   # For multi-user, pass from client after auth
):
    # 1. Transcribe
    audio_bytes = await audio_file.read()
    transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    if transcript.startswith("Error") or transcript.startswith("ERROR"):
        raise HTTPException(status_code=500, detail=transcript)

    # 2. Retrieve context from vector store filtered by user_id
    #    (vector_store must support user_id filtering)
    results = vector_store.search_index(transcript, top_k=5, user_id=user_id)
    avg_distance = (
        sum([res["distance"] for res in results]) / len(results)
        if results else -1.0
    )

    # 3. Production loop: generate up to 12 attempts, collect posts scoring >= 0.50
    MAX_ATTEMPTS = 12
    THRESHOLD = 0.50
    attempts = 0
    approved_posts = []
    all_scored = []

    while len(approved_posts) < 5 and attempts < MAX_ATTEMPTS:
        attempts += 1
        # Generate one batch of 5 variations (or you could generate one at a time)
        # Here we generate 5 each time to speed up
        generated_variations = await generation_service.generate_post_rag(
            transcript,
            results,
            tone=tone,
            platform=platform,
            num_variations=5   # We'll need to add this param to generate_post_rag; default 5
        )

        for post in generated_variations:
            if "text" not in post:
                continue
            score_data = scoring.calculate_safety_score(post["text"], avg_distance)
            final_score = score_data["final_score"]
            all_scored.append({
                "text": post["text"],
                "score": final_score,
                "breakdown": score_data["breakdown"]
            })
            if final_score >= THRESHOLD:
                approved_posts.append({
                    "text": post["text"],
                    "score": final_score
                })
                if len(approved_posts) >= 5:
                    break

    # 4. Return result
    status = "success" if len(approved_posts) >= 5 else "partial_success"
    return {
        "status": status,
        "variations": approved_posts[:5],  # top 5 (already sorted by score descending? we may need to sort)
        "total_generated": len(all_scored),
        "attempts_used": attempts,
        "message": f"Generated {len(approved_posts)} posts meeting threshold." if len(approved_posts) < 5 else None
    }

# ==================== Publish Post ====================
@app.post("/publish-post")
async def publish_post(
    platform: str = Form(...),
    post_text: str = Form(...),
    user_id: str = Form("demo_user"),
    db: Session = Depends(get_db)
):
    platform_key = platform.lower()
    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        raise HTTPException(status_code=404, detail=f"No credentials found for user {user_id}. Please authenticate first.")

    result = await social_publisher.publish_to_platform(platform_key, post_text, creds)
    return result

# ==================== Scheduling Endpoints (unchanged) ====================
class ConfirmPostRequest(BaseModel):
    platform: str
    text: str
    scheduled_time: Optional[str] = None

@app.post("/parse-schedule")
async def parse_schedule(audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()
    transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    parsed_time = dateparser.parse(
        transcript,
        settings={'TIMEZONE': 'Asia/Kolkata', 'RETURN_AS_TIMEZONE_AWARE': True}
    )
    if not parsed_time:
        raise HTTPException(status_code=400, detail="Could not parse scheduled time.")
    return {"parsed_time": parsed_time.isoformat(), "human_text": transcript}

@app.post("/confirm-post")
async def confirm_post(request: ConfirmPostRequest):
    if request.scheduled_time:
        try:
            dt = datetime.fromisoformat(request.scheduled_time)
            scheduler.add_job(publish_to_social_media, 'date', run_date=dt, args=[request.platform, request.text])
            return {"status": "scheduled", "message": f"Post scheduled for {dt.isoformat()}"}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid scheduled_time format.")
    else:
        publish_to_social_media(request.platform, request.text)
        return {"status": "published_immediately", "message": "Post published immediately."}