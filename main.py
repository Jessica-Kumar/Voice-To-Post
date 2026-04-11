import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
import tweepy
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import dateparser
from dateparser.search import search_dates
import PyPDF2
import io

import vector_store
import speech_service
import generation_service
import scoring
from database import get_db, SessionLocal, SocialCreds, encrypt_secret, decrypt_secret, download_db, upload_db
import social_publisher

load_dotenv()

app = FastAPI(title="Voice-To-Post Backend API")
scheduler = BackgroundScheduler()

LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")
TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET")
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

twitter_oauth_state = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    download_db()
    scheduler.start()
    sample_data = [
        "Welcome to Voice-To-Post backend!",
        "Vector databases help in doing semantic similarity search.",
        "FastAPI is a fast, highly performant web framework for building APIs."
    ]
    vector_store.add_text_to_index(sample_data, user_id="system")
    print("Application initialized.")

@app.get("/")
async def health_endpoint():
    return {"status": "Voice-To-Post Backend is running"}

# ==================== Bio Syncing Helpers ====================

async def sync_twitter_data(user_id: str, access_token: str, db: Session):
    try:
        # FIX: use access_token= not bearer_token= (bearer is app-level, not user-level)
        client = tweepy.Client(access_token=access_token)
        me = client.get_me(user_fields=["description"])
        if me.data:
            description = me.data.description
            creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
            if not creds:
                creds = SocialCreds(user_id=user_id)
                db.add(creds)
            creds.twitter_bio = description
            db.commit()
            if description:
                vector_store.add_text_to_index([description], user_id=user_id)
            print(f"Synced Twitter bio for user {user_id}")
    except Exception as e:
        print(f"Error syncing Twitter data: {e}")

async def sync_linkedin_data(user_id: str, access_token: str, db: Session):
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.linkedin.com/v2/userinfo", headers=headers)
        if resp.status_code != 200:
            print(f"LinkedIn userinfo error: {resp.status_code} - {resp.text}")
            return
        data = resp.json()
        name = data.get("name", "")
        bio = name or data.get("email", data.get("sub", ""))
        creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
        if not creds:
            creds = SocialCreds(user_id=user_id)
            db.add(creds)
        creds.linkedin_headline = bio
        db.commit()
        if bio:
            vector_store.add_text_to_index([bio], user_id=user_id)
        print(f"Synced LinkedIn bio for user {user_id}")

# ==================== OAuth Endpoints ====================

@app.get("/auth/linkedin/login")
async def linkedin_login():
    clean_base = BASE_URL.rstrip('/')
    redirect_uri = f"{clean_base}/auth/linkedin/callback"
    scope = "w_member_social,profile,openid"
    auth_url = (
        f"https://www.linkedin.com/oauth/v2/authorization"
        f"?response_type=code"
        f"&client_id={LINKEDIN_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope}"
    )
    return RedirectResponse(auth_url)

@app.get("/auth/linkedin/callback")
async def linkedin_callback(code: str, db: Session = Depends(get_db)):
    clean_base = BASE_URL.rstrip('/')
    redirect_uri = f"{clean_base}/auth/linkedin/callback"
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"LinkedIn token exchange failed: {resp.text}")
        token_data = resp.json()
        access_token = token_data["access_token"]

    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        userinfo = await client.get("https://api.linkedin.com/v2/userinfo", headers=headers)
        if userinfo.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not fetch user info")
        userinfo_data = userinfo.json()
        user_id = userinfo_data["sub"]

    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        creds = SocialCreds(user_id=user_id)
        db.add(creds)
    creds.linkedin_access_token = encrypt_secret(access_token)
    db.commit()
    upload_db()
    await sync_linkedin_data(user_id, access_token, db)

    # FIX: Added &platform=linkedin so Android app knows which platform connected
    return HTMLResponse(f"""
<html><body>
<h1>LinkedIn authentication successful!</h1>
<p>Your user ID: <strong>{user_id}</strong></p>
<p>You can close this window and return to the app.</p>
<script>window.location.href = "yourapp://callback?user_id={user_id}&platform=linkedin";</script>
</body></html>
""")

@app.get("/auth/twitter/login")
async def twitter_login():
    oauth2_handler = tweepy.OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        client_secret=TWITTER_CLIENT_SECRET,
        redirect_uri=f"{BASE_URL}/auth/twitter/callback",
        scope=["tweet.read", "tweet.write", "users.read", "offline.access"]
    )
    authorization_url, state = oauth2_handler.get_authorization_url()
    twitter_oauth_state[state] = {"code_verifier": oauth2_handler.code_verifier}
    return RedirectResponse(authorization_url)

@app.get("/auth/twitter/callback")
async def twitter_callback(code: str, state: str, db: Session = Depends(get_db)):
    stored = twitter_oauth_state.pop(state, None)
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    oauth2_handler = tweepy.OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        client_secret=TWITTER_CLIENT_SECRET,
        redirect_uri=f"{BASE_URL}/auth/twitter/callback",
        scope=["tweet.read", "tweet.write", "users.read", "offline.access"]
    )
    oauth2_handler.code_verifier = stored["code_verifier"]

    try:
        token_data = oauth2_handler.fetch_token(code)
        access_token = token_data["access_token"]
        refresh_token = token_data.get("refresh_token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Twitter token exchange failed: {str(e)}")

    client = tweepy.Client(access_token=access_token)
    me = client.get_me()
    if not me.data:
        raise HTTPException(status_code=400, detail="Could not fetch Twitter user info")
    user_id = str(me.data.id)

    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        creds = SocialCreds(user_id=user_id)
        db.add(creds)
    creds.twitter_access_token = encrypt_secret(access_token)
    if refresh_token:
        creds.twitter_refresh_token = encrypt_secret(refresh_token)
    db.commit()
    upload_db()
    await sync_twitter_data(user_id, access_token, db)

    # FIX: Added &platform=twitter so Android app knows which platform connected
    return HTMLResponse(f"""
<html><body>
<h1>Twitter authentication successful!</h1>
<p>Your user ID: <strong>{user_id}</strong></p>
<p>You can close this window and return to the app.</p>
<script>window.location.href = "yourapp://callback?user_id={user_id}&platform=twitter";</script>
</body></html>
""")

# ==================== Generation Endpoint (FIXED) ====================

@app.post("/generate-post")
async def generate_post(
    audio_file: UploadFile = File(...),
    tone: str = Form(...),
    platform: str = Form(...),
    user_id: str = Form(...)
):
    # ── Step 1: Transcribe ──────────────────────────────────────────────────
    audio_bytes = await audio_file.read()

    try:
        transcript = await asyncio.wait_for(
            speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type),
            timeout=20.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Speech-to-text timed out. Please try a shorter recording.")

    if transcript.startswith("Error") or transcript.startswith("ERROR"):
        raise HTTPException(status_code=500, detail=transcript)

    # ── Step 2: RAG context retrieval ───────────────────────────────────────
    results = vector_store.search_index(transcript, top_k=5, user_id=user_id)
    avg_distance = (
        sum(res["distance"] for res in results) / len(results)
        if results else -1.0
    )
    raw_context_text = " ".join(res["text"] for res in results) if results else ""

    # ── Step 3: ONE Gemini call — no retry loop ─────────────────────────────
    #
    # KEY FIX: The old code looped up to 15 times if posts didn't pass
    # a 0.75 threshold. Each Gemini call takes ~10s.
    # 15 × 10s = 150s → way over HuggingFace's ~60s connection timeout → spinner.
    #
    # New approach: call once, score all 5, return them sorted best→worst.
    # Total latency: ~15–20s. Always finishes. Users see differentiated scores.
    try:
        generated_variations = await asyncio.wait_for(
            generation_service.generate_post_rag(
                transcript,
                results,
                tone=tone,
                platform=platform,
                num_variations=5
            ),
            timeout=35.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Post generation timed out. Please try again.")

    # ── Step 4: Score all posts and sort ────────────────────────────────────
    scored_posts = []
    for post in generated_variations:
        if "text" not in post or not post["text"].strip():
            continue
        score_data = scoring.calculate_safety_score(
            generated_post=post["text"],
            context_distance=avg_distance,
            context_text=raw_context_text
        )
        scored_posts.append({
            "text": post["text"],
            "score": score_data["final_score"],
            "breakdown": score_data["breakdown"]
        })

    scored_posts.sort(key=lambda x: x["score"], reverse=True)

    if not scored_posts:
        raise HTTPException(status_code=500, detail="Generation returned no valid posts. Please try again.")

    return {
        "status": "success",
        "variations": scored_posts,
        "total_generated": len(scored_posts),
        "attempts_used": 1,
        "message": None
    }

# ==================== Publish Post ====================

@app.post("/publish-post")
async def publish_post(
    platform: str = Form(...),
    post_text: str = Form(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    post_text = post_text.replace("\\n", "\n")
    platform_key = platform.lower()
    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        raise HTTPException(status_code=404, detail=f"No credentials found for user {user_id}.")
    result = await social_publisher.publish_to_platform(platform_key, post_text, creds)
    if result and result.get("status") == "success":
        memory_text = f"[{platform_key.capitalize()} Post History]: {post_text}"
        vector_store.add_text_to_index([memory_text], user_id=user_id)
    return result

# ==================== Upload Brand Policy ====================

@app.post("/upload-policy")
async def upload_policy(
    user_id: str = Form(...),
    policy_file: UploadFile = File(...)
):
    filename = policy_file.filename.lower()
    content_bytes = await policy_file.read()
    extracted_text = ""
    try:
        if filename.endswith(".txt"):
            extracted_text = content_bytes.decode("utf-8")
        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        else:
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the file.")
        memory_text = f"[STRICT BRAND POLICY/GUIDELINE]: {extracted_text}"
        vector_store.add_text_to_index([memory_text], user_id=user_id)
        return {"status": "success", "message": f"Policy '{policy_file.filename}' successfully uploaded and memorized!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# ==================== Scheduling Endpoints ====================

@app.post("/parse-schedule")
async def parse_schedule(audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()
    transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    print(f"DEBUG - Scheduling Audio Transcript: '{transcript}'")
    found_dates = search_dates(
        transcript,
        settings={'TIMEZONE': 'Asia/Kolkata', 'RETURN_AS_TIMEZONE_AWARE': True}
    )
    if not found_dates:
        raise HTTPException(status_code=400, detail=f"Could not extract a valid time from the audio: '{transcript}'")
    parsed_time = found_dates[0][1]
    return {"parsed_time": parsed_time.isoformat(), "human_text": transcript}

class ConfirmPostRequest(BaseModel):
    platform: str
    text: str
    scheduled_time: Optional[str] = None
    user_id: str

def scheduled_publish_job(platform: str, text: str, user_id: str):
    db = SessionLocal()
    try:
        creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
        if not creds:
            print(f"[Scheduled Job] No credentials for user {user_id}")
            return
        result = asyncio.run(social_publisher.publish_to_platform(platform.lower(), text, creds))
        if result and result.get("status") == "success":
            memory_text = f"[{platform.capitalize()} Post History]: {text}"
            vector_store.add_text_to_index([memory_text], user_id=user_id)
    except Exception as e:
        print(f"[Scheduled Job] Error: {e}")
    finally:
        db.close()

@app.post("/confirm-post")
async def confirm_post(request: ConfirmPostRequest, db: Session = Depends(get_db)):
    if not request.scheduled_time:
        creds = db.query(SocialCreds).filter(SocialCreds.user_id == request.user_id).first()
        if not creds:
            raise HTTPException(status_code=404, detail="User credentials not found")
        result = await social_publisher.publish_to_platform(
            request.platform.lower(), request.text, creds
        )
        if result and result.get("status") == "success":
            memory_text = f"[{request.platform.capitalize()} Post History]: {request.text}"
            vector_store.add_text_to_index([memory_text], user_id=request.user_id)
        return {"status": "published_immediately", "result": result}

    try:
        dt = datetime.fromisoformat(request.scheduled_time)
        scheduler.add_job(
            scheduled_publish_job,
            'date',
            run_date=dt,
            args=[request.platform, request.text, request.user_id]
        )
        return {"status": "scheduled", "message": f"Post scheduled for {dt.isoformat()}"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scheduled_time format.")
