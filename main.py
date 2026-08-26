import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
import tweepy
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel
import dateparser
from dateparser.search import search_dates
import PyPDF2
import io

# Import existing services
import vector_store
import speech_service
import generation_service
import scoring
from database import get_db, SessionLocal, SocialCreds, ScheduledPost, encrypt_secret, decrypt_secret, download_db, upload_db, Base, engine
import social_publisher

# Import NEW services
import image_service
import auth_service
import thread_service
import refinement_service
from rate_limiter import limiter, RATE_LIMITS, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

load_dotenv()

app = FastAPI(
    title="Voice-To-Post Backend API v2.0",
    description="AI-powered voice-to-social-media platform with advanced features",
    version="2.0.0"
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

scheduler = BackgroundScheduler()

LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID")
TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

twitter_oauth_state = {}

# ==================== IMPROVED CORS Configuration ====================
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000,http://localhost:8080,http://localhost:5173").split(",")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS if ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global Exception Handler ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Standardized error response format"""
    error_type = type(exc).__name__

    # Don't expose internal errors in production
    if ENVIRONMENT == "production" and error_type not in ["HTTPException", "RateLimitExceeded"]:
        message = "An internal error occurred. Please try again later."
    else:
        message = str(exc)

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": error_type,
            "message": message,
            "endpoint": str(request.url.path)
        }
    )

# ==================== File Size Validation Middleware ====================
@app.middleware("http")
async def validate_file_size(request: Request, call_next):
    """Validate uploaded file sizes"""
    if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "error_type": "FileTooLarge",
                    "message": f"File too large. Maximum {MAX_FILE_SIZE // (1024*1024)}MB allowed.",
                    "max_size_mb": MAX_FILE_SIZE // (1024*1024)
                }
            )

    response = await call_next(request)
    return response

@app.on_event("startup")
async def startup_event():
    # Create all database tables (including new auth tables)
    Base.metadata.create_all(bind=engine)

    download_db()
    scheduler.start()
    restore_scheduled_jobs()
    print("✅ Voice-To-Post v2.0 initialized with all features!")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")

@app.on_event("shutdown")
async def shutdown_event():
    """Flush pending vector store changes to Hugging Face on shutdown."""
    try:
        vector_store.upload_vector_store_to_hf()
    except Exception as e:
        print(f"[Shutdown] Vector store upload failed: {e}")

@app.get("/")
@limiter.limit(RATE_LIMITS["general"])
async def health_endpoint(request: Request):
    return {
        "status": "Voice-To-Post Backend v2.0 is running",
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
        "features": [
            "Voice-to-Post Generation",
            "Multi-Platform Publishing",
            "AI Image Generation (FREE)",
            "Thread Generator",
            "Post Refinement",
            "Smart Hashtags",
            "Rate Limiting",
            "JWT Authentication",
            "Vector Store Persistence"
        ]
    }

# ==================== NEW: Authentication Endpoints ====================

@app.post("/auth/register")
@limiter.limit(RATE_LIMITS["auth"])
async def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Register a new user account."""
    try:
        user = auth_service.create_user(db, email, password, full_name)

        # Persist users table to HF so registrations survive Space restarts
        upload_db()

        # Create access token
        token = auth_service.create_access_token({"sub": user.user_id})

        return {
            "status": "success",
            "user_id": user.user_id,
            "email": user.email,
            "access_token": token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
@limiter.limit(RATE_LIMITS["auth"])
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Login and get access token."""
    user = auth_service.authenticate_user(db, email, password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create access token
    token = auth_service.create_access_token({"sub": user.user_id})

    return {
        "status": "success",
        "user_id": user.user_id,
        "email": user.email,
        "access_token": token,
        "token_type": "bearer"
    }

@app.post("/auth/create-api-key")
@limiter.limit(RATE_LIMITS["auth"])
async def create_api_key(
    request: Request,
    name: Optional[str] = Form(None),
    current_user: auth_service.User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db)
):
    """Create an API key for programmatic access."""
    api_key = auth_service.create_api_key(db, current_user.user_id, name)

    # Persist API keys so they survive Space restarts
    upload_db()

    return {
        "status": "success",
        "api_key": api_key.api_key,
        "name": api_key.name,
        "created_at": api_key.created_at.isoformat(),
        "message": "Save this API key securely. You won't be able to see it again."
    }

@app.get("/auth/me")
@limiter.limit(RATE_LIMITS["general"])
async def get_current_user_info(
    request: Request,
    current_user: auth_service.User = Depends(auth_service.get_current_user)
):
    """Get current authenticated user information."""
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }

# ==================== Bio Syncing Helpers ====================

async def sync_twitter_data(user_id: str, access_token: str, db: Session):
    try:
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

# ==================== OAuth Endpoints (Existing) ====================

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
    # Cap pending states to avoid unbounded memory growth from abandoned flows
    if len(twitter_oauth_state) > 200:
        for old_state in list(twitter_oauth_state.keys())[:100]:
            twitter_oauth_state.pop(old_state, None)
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

    return HTMLResponse(f"""
<html><body>
<h1>Twitter authentication successful!</h1>
<p>Your user ID: <strong>{user_id}</strong></p>
<p>You can close this window and return to the app.</p>
<script>window.location.href = "yourapp://callback?user_id={user_id}&platform=twitter";</script>
</body></html>
""")

@app.get("/auth/discord/login")
async def discord_login():
    clean_base = BASE_URL.rstrip('/')
    redirect_uri = f"{clean_base}/auth/discord/callback"
    scope = "webhook.incoming identify"
    auth_url = (
        f"https://discord.com/api/oauth2/authorize"
        f"?client_id={DISCORD_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope={scope}"
    )
    return RedirectResponse(auth_url)

@app.get("/auth/discord/callback")
async def discord_callback(code: str, db: Session = Depends(get_db)):
    clean_base = BASE_URL.rstrip('/')
    redirect_uri = f"{clean_base}/auth/discord/callback"
    token_url = "https://discord.com/api/oauth2/token"

    data = {
        "client_id": DISCORD_CLIENT_ID,
        "client_secret": DISCORD_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Discord token exchange failed: {resp.text}")

        token_data = resp.json()
        access_token = token_data.get("access_token")
        webhook_url = token_data.get("webhook", {}).get("url")

        if not webhook_url:
            raise HTTPException(status_code=400, detail="No webhook URL returned from Discord. User may not have authorized a channel.")

    user_headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        user_resp = await client.get("https://discord.com/api/users/@me", headers=user_headers)
        if user_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not fetch Discord user info")

        user_id = str(user_resp.json()["id"])

    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        creds = SocialCreds(user_id=user_id)
        db.add(creds)

    creds.discord_webhook_url = encrypt_secret(webhook_url)
    db.commit()
    upload_db()

    return HTMLResponse(f"""
<html><body>
<h1>Discord authentication successful!</h1>
<p>Your user ID: <strong>{user_id}</strong></p>
<p>You can close this window and return to the app.</p>
<script>window.location.href = "yourapp://callback?user_id={user_id}&platform=discord";</script>
</body></html>
""")

@app.post("/auth/save-tokens")
@limiter.limit(RATE_LIMITS["general"])
async def save_manual_tokens(
    request: Request,
    user_id: str = Form(...),
    discord_webhook_url: Optional[str] = Form(None),
    medium_integration_token: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Save manual tokens for a user."""
    creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
    if not creds:
        creds = SocialCreds(user_id=user_id)
        db.add(creds)

    if discord_webhook_url:
        creds.discord_webhook_url = encrypt_secret(discord_webhook_url)
    if medium_integration_token:
        creds.medium_integration_token = encrypt_secret(medium_integration_token)

    db.commit()
    upload_db()

    return {"status": "success", "message": "Tokens saved successfully."}

# ==================== Generation Endpoint (Enhanced) ====================

async def _generate_post_core(
    audio_file: UploadFile,
    tone: str,
    platform: str,
    user_id: str
) -> dict:
    """Shared voice-to-post pipeline used by all generation endpoints."""
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

    results = vector_store.search_index(transcript, top_k=5, user_id=user_id)
    avg_distance = (
        sum(res["distance"] for res in results) / len(results)
        if results else -1.0
    )
    raw_context_text = " ".join(res["text"] for res in results) if results else ""

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
        "transcript": transcript,
        "variations": scored_posts,
        "total_generated": len(scored_posts),
        "attempts_used": 1,
        "message": None
    }


@app.post("/generate-post")
@limiter.limit(RATE_LIMITS["generation"])
async def generate_post(
    request: Request,
    audio_file: UploadFile = File(...),
    tone: str = Form(...),
    platform: str = Form(...),
    user_id: str = Form(...)
):
    """Generate post variations from voice recording."""
    return await _generate_post_core(audio_file, tone, platform, user_id)

# ==================== NEW: Image Generation Endpoints ====================

@app.post("/generate-post-with-image")
@limiter.limit(RATE_LIMITS["generation"])
async def generate_post_with_image(
    request: Request,
    audio_file: UploadFile = File(...),
    tone: str = Form(...),
    platform: str = Form(...),
    user_id: str = Form(...),
    image_method: str = Form("stock"),  # "stock" or "ai"
    num_image_options: int = Form(3),
    return_base64: bool = Form(True)  # NEW: Option to skip base64 encoding
):
    """Generate post WITH image options."""
    # First generate post
    post_response = await _generate_post_core(audio_file, tone, platform, user_id)

    # Get best post
    best_post = post_response["variations"][0]["text"]

    # Generate image options
    try:
        if num_image_options > 1:
            images = await image_service.get_multiple_image_options(
                best_post,
                platform,
                num_image_options
            )
        else:
            image = await image_service.generate_image_from_post(
                best_post,
                platform,
                image_method
            )
            images = [image]

        # Encode images as base64 for response (optional)
        for img in images:
            if img.get("image_bytes"):
                if return_base64:
                    img["image_base64"] = image_service.encode_image_base64(img["image_bytes"])
                del img["image_bytes"]  # Remove bytes from response

        return {
            **post_response,
            "images": images,
            "image_count": len(images)
        }
    except Exception as e:
        # Return post without images if image generation fails
        return {
            **post_response,
            "images": [],
            "image_error": str(e),
            "message": "Post generated successfully but image generation failed. You can still use the post."
        }

@app.post("/generate-image-for-post")
@limiter.limit(RATE_LIMITS["generation"])
async def generate_image_for_post(
    request: Request,
    post_text: str = Form(...),
    platform: str = Form("twitter"),
    method: str = Form("stock"),
    num_options: int = Form(3),
    return_base64: bool = Form(True)  # NEW: Option to skip base64 encoding
):
    """Generate images for an existing post."""
    try:
        if num_options > 1:
            images = await image_service.get_multiple_image_options(
                post_text,
                platform,
                num_options
            )
        else:
            image = await image_service.generate_image_from_post(
                post_text,
                platform,
                method
            )
            images = [image]

        # Encode as base64 (optional)
        for img in images:
            if img.get("image_bytes"):
                if return_base64:
                    img["image_base64"] = image_service.encode_image_base64(img["image_bytes"])
                del img["image_bytes"]

        return {
            "status": "success",
            "images": images,
            "count": len(images)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW: Thread Generator ====================

@app.post("/generate-thread")
@limiter.limit(RATE_LIMITS["generation"])
async def generate_thread_endpoint(
    request: Request,
    audio_file: UploadFile = File(...),
    platform: str = Form(...),
    tone: str = Form(...),
    user_id: str = Form(...),
    max_posts: int = Form(5)
):
    """Generate a multi-post thread from voice recording."""
    audio_bytes = await audio_file.read()

    try:
        transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # Get user context
    results = vector_store.search_index(transcript, top_k=3, user_id=user_id)
    context = " ".join(res["text"] for res in results) if results else ""

    # Generate thread
    try:
        thread = await thread_service.generate_thread(
            transcript=transcript,
            platform=platform,
            tone=tone,
            max_posts=max_posts,
            context=context
        )

        return {
            "status": "success",
            "transcript": transcript,
            "thread": thread,
            "total_posts": len(thread),
            "platform": platform
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW: Multi-Platform Generator ====================

@app.post("/generate-cross-platform")
@limiter.limit(RATE_LIMITS["generation"])
async def generate_cross_platform(
    request: Request,
    audio_file: UploadFile = File(...),
    platforms: str = Form(...),  # Comma-separated: "twitter,linkedin,discord"
    tone: str = Form(...),
    user_id: str = Form(...)
):
    """Generate optimized posts for multiple platforms at once."""
    audio_bytes = await audio_file.read()

    try:
        transcript = await speech_service.transcribe_audio_bytes(audio_bytes, audio_file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # Parse platforms
    platform_list = [p.strip() for p in platforms.split(",")]

    # Get user context
    results = vector_store.search_index(transcript, top_k=3, user_id=user_id)
    context = " ".join(res["text"] for res in results) if results else ""

    # Generate for all platforms
    try:
        posts = await thread_service.generate_multi_platform_posts(
            transcript=transcript,
            platforms=platform_list,
            tone=tone,
            context=context
        )

        return {
            "status": "success",
            "transcript": transcript,
            "posts": posts,
            "platforms": list(posts.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW: Post Refinement ====================

@app.post("/refine-post")
@limiter.limit(RATE_LIMITS["general"])
async def refine_post_endpoint(
    request: Request,
    post_text: str = Form(...),
    refinement_type: str = Form(...),
    platform: str = Form("twitter"),
    custom_instruction: Optional[str] = Form(None)
):
    """Refine an existing post (shorten, lengthen, change tone, etc.)."""
    try:
        refined = await refinement_service.refine_post(
            original_post=post_text,
            refinement_type=refinement_type,
            platform=platform,
            custom_instruction=custom_instruction
        )

        return {
            "status": "success",
            "original": post_text,
            "refined": refined,
            "refinement_type": refinement_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/refinement-types")
@limiter.limit(RATE_LIMITS["general"])
async def get_refinement_types(request: Request):
    """Get available refinement types."""
    return {
        "refinement_types": refinement_service.AVAILABLE_REFINEMENTS
    }

@app.post("/analyze-post")
@limiter.limit(RATE_LIMITS["analytics"])
async def analyze_post_endpoint(
    request: Request,
    post_text: str = Form(...),
    platform: str = Form("twitter")
):
    """Analyze post quality and get improvement suggestions."""
    try:
        analysis = await refinement_service.analyze_post_quality(post_text, platform)
        return {
            "status": "success",
            "post": post_text,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW: Smart Hashtag Suggestions ====================

@app.post("/suggest-hashtags")
@limiter.limit(RATE_LIMITS["general"])
async def suggest_hashtags_endpoint(
    request: Request,
    post_text: str = Form(...),
    platform: str = Form("twitter"),
    num_hashtags: int = Form(5)
):
    """Get smart hashtag suggestions for a post."""
    try:
        hashtags = await thread_service.smart_hashtag_suggestions(
            post_text,
            platform,
            num_hashtags
        )

        return {
            "status": "success",
            "hashtags": hashtags,
            "formatted": ["#" + tag for tag in hashtags]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Publish Post ====================

@app.post("/publish-post")
@limiter.limit(RATE_LIMITS["publish"])
async def publish_post(
    request: Request,
    platform: str = Form(...),
    post_text: str = Form(...),
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
@limiter.limit(RATE_LIMITS["upload"])
async def upload_policy(
    request: Request,
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
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")

# ==================== Scheduling Endpoints ====================

@app.post("/parse-schedule")
@limiter.limit(RATE_LIMITS["general"])
async def parse_schedule(request: Request, audio_file: UploadFile = File(...)):
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

def scheduled_publish_job(platform: str, text: str, user_id: str, scheduled_row_id: int = None):
    db = SessionLocal()
    try:
        creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()
        if not creds:
            print(f"[Scheduled Job] No credentials for user {user_id}")
            result_status = "failed"
        else:
            result = asyncio.run(social_publisher.publish_to_platform(platform.lower(), text, creds))
            result_status = "published" if result and result.get("status") == "success" else "failed"
            if result_status == "published":
                memory_text = f"[{platform.capitalize()} Post History]: {text}"
                vector_store.add_text_to_index([memory_text], user_id=user_id)

        # Mark the persisted row so restarts don't re-publish it
        if scheduled_row_id is not None:
            row = db.query(ScheduledPost).filter(ScheduledPost.id == scheduled_row_id).first()
            if row:
                row.status = result_status
                db.commit()
                upload_db()
    except Exception as e:
        print(f"[Scheduled Job] Error: {e}")
        if scheduled_row_id is not None:
            try:
                row = db.query(ScheduledPost).filter(ScheduledPost.id == scheduled_row_id).first()
                if row:
                    row.status = "failed"
                    db.commit()
                    upload_db()
            except Exception:
                pass
    finally:
        db.close()


def restore_scheduled_jobs():
    """Re-register pending scheduled posts after a restart (HF Spaces restart often)."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        rows = db.query(ScheduledPost).filter(ScheduledPost.status == "pending").all()
        restored = 0
        for row in rows:
            try:
                run_at = datetime.fromisoformat(row.scheduled_time)
                if run_at.tzinfo is None:
                    run_at = run_at.replace(tzinfo=timezone.utc)
            except ValueError:
                row.status = "failed"
                continue
            # Publish missed jobs immediately, future jobs at their scheduled time
            scheduler.add_job(
                scheduled_publish_job,
                'date',
                run_date=max(run_at, now),
                args=[row.platform, row.text, row.user_id, row.id]
            )
            restored += 1
        db.commit()
        if restored:
            print(f"✅ Restored {restored} scheduled post(s) after restart")
    except Exception as e:
        print(f"[Startup] Error restoring scheduled jobs: {e}")
    finally:
        db.close()

@app.post("/confirm-post")
@limiter.limit(RATE_LIMITS["publish"])
async def confirm_post(request: Request, post_request: ConfirmPostRequest, db: Session = Depends(get_db)):
    if not post_request.scheduled_time:
        creds = db.query(SocialCreds).filter(SocialCreds.user_id == post_request.user_id).first()
        if not creds:
            raise HTTPException(status_code=404, detail="User credentials not found")
        result = await social_publisher.publish_to_platform(
            post_request.platform.lower(), post_request.text, creds
        )
        if result and result.get("status") == "success":
            memory_text = f"[{post_request.platform.capitalize()} Post History]: {post_request.text}"
            vector_store.add_text_to_index([memory_text], user_id=post_request.user_id)
        return {"status": "published_immediately", "result": result}

    try:
        dt = datetime.fromisoformat(post_request.scheduled_time)
        # Persist the schedule so it survives Space restarts
        row = ScheduledPost(
            platform=post_request.platform.lower(),
            text=post_request.text,
            user_id=post_request.user_id,
            scheduled_time=dt.isoformat(),
            status="pending"
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        upload_db()
        scheduler.add_job(
            scheduled_publish_job,
            'date',
            run_date=dt,
            args=[post_request.platform, post_request.text, post_request.user_id, row.id]
        )
        return {"status": "scheduled", "message": f"Post scheduled for {dt.isoformat()}", "schedule_id": row.id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scheduled_time format.")

# ==================== NEW: Vector Store Stats ====================

@app.get("/vector-store/stats")
@limiter.limit(RATE_LIMITS["general"])
async def get_vector_store_stats(request: Request):
    """Get vector store statistics."""
    stats = vector_store.get_vector_store_stats()
    return {
        "status": "success",
        "stats": stats
    }

# ==================== NEW: System Info ====================

@app.get("/system/info")
@limiter.limit(RATE_LIMITS["general"])
async def system_info(request: Request):
    """Get system information and feature availability."""
    return {
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
        "features": {
            "voice_to_post": True,
            "image_generation": {
                "available": bool(os.getenv("PEXELS_API_KEY") or os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("HF_TOKEN")),
                "methods": ["stock", "ai"]
            },
            "thread_generation": True,
            "post_refinement": True,
            "smart_hashtags": True,
            "multi_platform": True,
            "authentication": True,
            "rate_limiting": True,
            "vector_persistence": True
        },
        "rate_limits": RATE_LIMITS,
        "supported_platforms": ["twitter", "linkedin", "discord", "medium"],
        "supported_languages": "36+ via Deepgram"
    }
