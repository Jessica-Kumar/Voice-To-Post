# Voice-To-Post: Complete Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Core Components](#core-components)
5. [Features](#features)
6. [API Endpoints](#api-endpoints)
7. [Data Flow](#data-flow)
8. [Security & Authentication](#security--authentication)
9. [Setup & Deployment](#setup--deployment)
10. [Environment Variables](#environment-variables)

---

## 🎯 Project Overview

**Voice-To-Post** is an AI-powered social media content generation platform that converts voice recordings into platform-optimized social media posts. The system uses advanced speech-to-text, RAG (Retrieval-Augmented Generation), and LLM technologies to create contextually relevant, brand-aligned posts across multiple social platforms.

### Key Capabilities
- 🎤 Voice-to-text transcription using Deepgram Nova-3
- 🤖 AI-powered post generation with Google Gemini 2.5 Flash
- 📊 RAG-based context retrieval for personalized content
- 🎯 Multi-platform support (Twitter, LinkedIn, Discord, Medium)
- 📈 Intelligent scoring system for post quality
- ⏰ Post scheduling capabilities
- 🔒 Encrypted credential storage
- 📄 Brand policy document integration

---

## 🏗️ Architecture

### System Design
```
┌─────────────────┐
│   Mobile App    │ (Android/iOS)
│   or Frontend   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│          FastAPI Backend (main.py)              │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐            │
│  │   OAuth      │  │   Upload     │            │
│  │   Handlers   │  │   Endpoints  │            │
│  └──────┬───────┘  └──────┬───────┘            │
│         │                  │                     │
│  ┌──────▼──────────────────▼──────┐            │
│  │     Speech Service              │            │
│  │   (Deepgram Nova-3)             │            │
│  └──────┬──────────────────────────┘            │
│         │                                        │
│  ┌──────▼──────────────────────────┐            │
│  │     Vector Store (FAISS)        │            │
│  │   (SentenceTransformer)         │            │
│  └──────┬──────────────────────────┘            │
│         │                                        │
│  ┌──────▼──────────────────────────┐            │
│  │  Generation Service             │            │
│  │   (Google Gemini + RAG)         │            │
│  └──────┬──────────────────────────┘            │
│         │                                        │
│  ┌──────▼──────────────────────────┐            │
│  │    Scoring Engine                │            │
│  │  (Quality Assessment)            │            │
│  └──────┬──────────────────────────┘            │
│         │                                        │
│  ┌──────▼──────────────────────────┐            │
│  │   Social Publisher               │            │
│  │  (Multi-platform posting)        │            │
│  └──────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         Data Layer                              │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐            │
│  │   SQLite DB  │  │  Hugging Face│            │
│  │  (Encrypted) │  │  Hub Storage │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend Framework
- **FastAPI**: High-performance async web framework
- **Uvicorn**: ASGI server for FastAPI
- **Python 3.10**: Core programming language

### AI & Machine Learning
- **Deepgram Nova-3**: Advanced speech-to-text transcription
- **Google Gemini 2.5 Flash**: LLM for content generation
- **LangChain**: LLM orchestration framework
- **Sentence Transformers** (`all-MiniLM-L6-v2`): Text embeddings
- **FAISS**: Vector similarity search (Facebook AI)
- **NewsAPI**: Real-time news enrichment (optional)

### Database & Storage
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Local database with WAL mode
- **Hugging Face Hub**: Cloud persistence layer
- **Cryptography (Fernet)**: Symmetric encryption for credentials

### Social Media APIs
- **Tweepy**: Twitter/X API integration
- **LinkedIn API**: Professional network posting
- **Discord Webhooks**: Community posting
- **Medium API**: Blog post publishing

### Utilities
- **APScheduler**: Post scheduling system
- **dateparser**: Natural language date parsing
- **PyPDF2**: PDF document processing
- **httpx**: Async HTTP client

---

## 🧩 Core Components

### 1. Speech Service (`speech_service.py`)
**Purpose**: Convert audio recordings to text

**Key Features**:
- Async audio transcription via Deepgram REST API
- Nova-3 model with smart formatting
- Supports multiple audio formats
- 60-second timeout for processing
- Robust error handling

**Main Function**:
```python
async def transcribe_audio_bytes(audio_bytes: bytes, content_type: str) -> str
```

---

### 2. Vector Store (`vector_store.py`)
**Purpose**: Semantic search and context retrieval

**Technology**:
- **Embeddings**: SentenceTransformer `all-MiniLM-L6-v2` (384 dimensions)
- **Index**: FAISS FlatL2 (L2 distance metric)
- **Storage**: In-memory with user-scoped filtering

**Key Functions**:
```python
def add_text_to_index(text_list: List[str], user_id: str) -> None
def search_index(query_text: str, top_k: int = 3, user_id: str = None) -> List[Dict]
```

**What Gets Indexed**:
- User bios from Twitter/LinkedIn
- Previously published posts
- Brand policy documents
- System seed data

**Search Strategy**:
- Retrieves top 50 candidates
- Filters by user_id
- Returns top K most relevant (default 3)

---

### 3. Generation Service (`generation_service.py`)
**Purpose**: Generate platform-optimized social media posts

**LLM Configuration**:
- **Model**: Gemini 2.5 Flash
- **Temperature**: 0.2 (stable, consistent output)
- **Top-P**: 0.1 (deterministic sampling)

**Generation Pipeline**:
1. Receive transcript + RAG context
2. Optional NewsAPI enrichment
3. Format context with vector store results
4. Execute strict prompt template
5. Parse JSON output (5 variations)
6. Replace escaped newlines with actual line breaks

**Anti-Hallucination Rules**:
- Zero fabrication of facts
- Ghostwriting mode (matches user's voice)
- Disconnect fallback (no forced connections)
- No generic AI buzzwords
- Platform-specific constraints

**Output Format**:
```json
[
  {"text": "Post variation 1"},
  {"text": "Post variation 2"},
  ...
]
```

---

### 4. Scoring Engine (`scoring.py`)
**Purpose**: Evaluate post quality with multi-factor scoring

**Scoring Formula**:
```
Final Score = 0.3×AI_Confidence + 0.3×Retrieval_Relevance + 0.3×Safety + 0.1×Engagement
```

#### Factors:

**1. AI Confidence (30%)**
- Post length optimization
- Sentence structure quality
- Twitter: 60-280 chars optimal
- LinkedIn: 280-1500 chars optimal

**2. Retrieval Relevance (30%)**
- Word overlap with context
- Vector distance factor
- Rewards context-grounded posts

**3. Safety Score (30%)**
- Forbidden term detection
- Length boundary checks
- Penalizes spam indicators

**4. Engagement Potential (10%)**
- Hashtag count (1-3 optimal)
- Emoji usage (1-3 optimal)
- Call-to-action words
- Question/exclamation marks

**Output**:
```python
{
  "final_score": 0.847,
  "breakdown": {
    "ai_confidence": 0.9,
    "retrieval_relevance": 0.82,
    "safety_score": 1.0,
    "engagement_potential": 0.65
  }
}
```

---

### 5. Social Publisher (`social_publisher.py`)
**Purpose**: Publish content to social platforms

**Supported Platforms**:

#### Twitter/X
- OAuth 2.0 with user access tokens
- 280 character limit
- Returns tweet URL

#### LinkedIn
- OpenID Connect authentication
- UGC Posts API
- Professional formatting
- Public visibility

#### Discord
- Webhook-based posting
- Channel-specific webhooks
- No OAuth required

#### Medium
- Integration token authentication
- Markdown content format
- Auto-extracts title from first line
- Public/draft status options

**Main Function**:
```python
async def publish_to_platform(platform: str, post_content: str, creds) -> dict
```

---

### 6. Database (`database.py`)
**Purpose**: Secure credential storage and cloud persistence

**Schema** (`SocialCreds`):
```python
- id: Integer (Primary Key)
- user_id: String (Unique, Indexed)
- twitter_access_token: String (Encrypted)
- twitter_refresh_token: String (Encrypted)
- twitter_bio: String
- linkedin_access_token: String (Encrypted)
- linkedin_vanity_name: String
- linkedin_headline: String
- discord_webhook_url: String (Encrypted)
- medium_integration_token: String (Encrypted)
```

**Security Features**:
- Fernet symmetric encryption
- Environment-based encryption key
- WAL journal mode for cloud compatibility
- Automatic Hugging Face sync

**Cloud Persistence**:
- Downloads DB on startup from HF Dataset
- Uploads after credential changes
- Dataset: `JessicaKumar/voice-to-post-data`

---

### 7. Main Application (`main.py`)
**Purpose**: FastAPI application orchestration

**Key Responsibilities**:
- Endpoint routing
- OAuth flow management
- Request/response handling
- Scheduler management
- CORS configuration
- Database session management

---

## ✨ Features

### 1. Voice-to-Post Generation
- Upload audio recording
- Automatic transcription
- Context-aware post generation
- 5 scored variations per request
- Platform-specific formatting

### 2. Multi-Platform Support
- **Twitter/X**: Short-form, punchy tweets
- **LinkedIn**: Professional, detailed posts
- **Discord**: Community announcements
- **Medium**: Long-form articles

### 3. RAG-Powered Personalization
- Learns from user's social profiles
- Remembers past posts
- Brand policy adherence
- Semantic similarity search

### 4. Intelligent Scoring
- Quality assessment for each variation
- Best post surfaces first
- Transparent scoring breakdown
- Multi-factor evaluation

### 5. OAuth Authentication
- Secure platform connections
- Token encryption
- Refresh token support (Twitter)
- Deep link callbacks for mobile

### 6. Post Scheduling
- Natural language time parsing
- Voice-based schedule input
- Background job execution
- APScheduler integration

### 7. Brand Policy Upload
- PDF/TXT document support
- Automatic text extraction
- Policy enforcement in generation
- Vector-indexed for retrieval

### 8. Post History Memory
- Tracks published content
- Improves future suggestions
- User-scoped context
- Style consistency

---

## 📡 API Endpoints

### Health Check
```http
GET /
```
Returns API status.

---

### OAuth Flows

#### LinkedIn
```http
GET /auth/linkedin/login
GET /auth/linkedin/callback?code={code}
```

#### Twitter
```http
GET /auth/twitter/login
GET /auth/twitter/callback?code={code}&state={state}
```

#### Discord
```http
GET /auth/discord/login
GET /auth/discord/callback?code={code}
```

#### Manual Token Save
```http
POST /auth/save-tokens
Content-Type: multipart/form-data

user_id: string
discord_webhook_url?: string
medium_integration_token?: string
```

---

### Post Generation

```http
POST /generate-post
Content-Type: multipart/form-data

audio_file: File (required)
tone: string (required) - "professional", "casual", "funny", etc.
platform: string (required) - "twitter", "linkedin", "discord", "medium"
user_id: string (required)
```

**Response**:
```json
{
  "status": "success",
  "variations": [
    {
      "text": "Post content here...",
      "score": 0.847,
      "breakdown": {
        "ai_confidence": 0.9,
        "retrieval_relevance": 0.82,
        "safety_score": 1.0,
        "engagement_potential": 0.65
      }
    }
  ],
  "total_generated": 5,
  "attempts_used": 1,
  "message": null
}
```

**Timeouts**:
- Transcription: 20 seconds
- Generation: 35 seconds
- Total: ~15-20 seconds typical

---

### Post Publishing

```http
POST /publish-post
Content-Type: multipart/form-data

platform: string (required)
post_text: string (required)
user_id: string (required)
```

**Response**:
```json
{
  "status": "success",
  "platform": "twitter",
  "post_id": "1234567890",
  "url": "https://twitter.com/user/status/1234567890",
  "message": "Successfully posted to Twitter!"
}
```

---

### Brand Policy Upload

```http
POST /upload-policy
Content-Type: multipart/form-data

user_id: string (required)
policy_file: File (required) - .txt or .pdf
```

**Response**:
```json
{
  "status": "success",
  "message": "Policy 'brand_guidelines.pdf' successfully uploaded and memorized!"
}
```

---

### Scheduling

#### Parse Schedule from Voice
```http
POST /parse-schedule
Content-Type: multipart/form-data

audio_file: File (required)
```

**Response**:
```json
{
  "parsed_time": "2026-08-23T15:30:00+05:30",
  "human_text": "tomorrow at three thirty pm"
}
```

#### Confirm Post (Immediate or Scheduled)
```http
POST /confirm-post
Content-Type: application/json

{
  "platform": "twitter",
  "text": "Post content",
  "scheduled_time": "2026-08-23T15:30:00" (optional),
  "user_id": "user123"
}
```

**Immediate Response**:
```json
{
  "status": "published_immediately",
  "result": { /* publish result */ }
}
```

**Scheduled Response**:
```json
{
  "status": "scheduled",
  "message": "Post scheduled for 2026-08-23T15:30:00"
}
```

---

## 🔄 Data Flow

### Complete User Journey

```
1. USER AUTHENTICATION
   ┌─────────────────────────────┐
   │ User clicks "Connect Twitter"│
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ OAuth flow initiated        │
   │ → Redirect to platform      │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ User authorizes             │
   │ → Callback with auth code   │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Exchange code for token     │
   │ → Encrypt and store in DB   │
   │ → Upload DB to HF Hub       │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Sync profile data           │
   │ → Fetch bio/headline        │
   │ → Index in vector store     │
   └─────────────────────────────┘

2. POST GENERATION
   ┌─────────────────────────────┐
   │ User records voice message  │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Upload audio + metadata     │
   │ → POST /generate-post       │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Transcription (Deepgram)    │
   │ → Audio bytes → text        │
   │ → ~2-5 seconds              │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ RAG Context Retrieval       │
   │ → Embed transcript          │
   │ → Search FAISS index        │
   │ → Return top 5 matches      │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Optional News Enrichment    │
   │ → Query NewsAPI             │
   │ → Add relevant headlines    │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ LLM Generation (Gemini)     │
   │ → Format prompt             │
   │ → Generate 5 variations     │
   │ → ~10-15 seconds            │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Scoring Engine              │
   │ → Score each variation      │
   │ → Calculate 4 factors       │
   │ → Sort by final score       │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Return to user              │
   │ → Best post first           │
   │ → Show score breakdown      │
   └─────────────────────────────┘

3. POST PUBLISHING
   ┌─────────────────────────────┐
   │ User selects post           │
   │ → Choose platform           │
   │ → Optional: set schedule    │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ POST /confirm-post          │
   └─────────────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   [Immediate]       [Scheduled]
        │                 │
        ▼                 ▼
   ┌─────────┐     ┌──────────────┐
   │ Publish │     │ Add APScheduler│
   │ Now     │     │ Job           │
   └────┬────┘     └──────┬───────┘
        │                 │
        └────────┬────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Social Publisher            │
   │ → Decrypt credentials       │
   │ → Call platform API         │
   │ → Return result             │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Update Vector Store         │
   │ → Index published post      │
   │ → Improve future context    │
   └─────────────────────────────┘
```

---

## 🔒 Security & Authentication

### Encryption
- **Algorithm**: Fernet (symmetric encryption)
- **Key Management**: Environment variable `ENCRYPTION_KEY`
- **Encrypted Fields**: All OAuth tokens and webhooks
- **Key Generation**:
```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

### OAuth Flows

#### Twitter OAuth 2.0
- **Grant Type**: Authorization Code with PKCE
- **Scopes**: `tweet.read`, `tweet.write`, `users.read`, `offline.access`
- **Token Refresh**: Supported via refresh_token
- **Callback**: `yourapp://callback?user_id={id}&platform=twitter`

#### LinkedIn OAuth 2.0
- **Grant Type**: Authorization Code
- **Scopes**: `w_member_social`, `profile`, `openid`
- **User Identification**: OpenID Connect `/userinfo`
- **Callback**: `yourapp://callback?user_id={id}&platform=linkedin`

#### Discord OAuth 2.0
- **Grant Type**: Authorization Code
- **Scopes**: `webhook.incoming`, `identify`
- **No Posting Token**: Uses webhook URL instead
- **Callback**: `yourapp://callback?user_id={id}&platform=discord`

#### Medium
- **Authentication**: Integration Token (self-service)
- **No OAuth Flow**: User manually provides token
- **Saved via**: `/auth/save-tokens` endpoint

### Database Security
- **SQLite WAL Mode**: Cloud-friendly journaling
- **Timeout**: 30 seconds for locked operations
- **Permissions**: 0777 on `/tmp/` for HF Spaces
- **Backup**: Automatic sync to Hugging Face Hub

### API Security Considerations
- **CORS**: Wide-open (`allow_origins=["*"]`) - should be restricted in production
- **No Rate Limiting**: Should be added for production
- **No API Keys**: Endpoints are open - should add authentication
- **Secrets in Env**: Proper secret management via environment variables

---

## 🚀 Setup & Deployment

### Local Development

1. **Clone Repository**
```bash
git clone <repo-url>
cd Voice-To-Post
```

2. **Create Virtual Environment**
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Environment Variables**
Create a `.env` file (see Environment Variables section)

5. **Run Application**
```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

6. **Test API**
```bash
curl http://localhost:7860/
```

---

### Docker Deployment

1. **Build Image**
```bash
docker build -t voice-to-post .
```

2. **Run Container**
```bash
docker run -p 7860:7860 \
  -e ENCRYPTION_KEY="your-key" \
  -e DEEPGRAM_API_KEY="your-key" \
  -e GEMINI_API_KEY="your-key" \
  -e TWITTER_CLIENT_ID="your-id" \
  -e TWITTER_CLIENT_SECRET="your-secret" \
  -e LINKEDIN_CLIENT_ID="your-id" \
  -e LINKEDIN_CLIENT_SECRET="your-secret" \
  -e HF_TOKEN="your-token" \
  voice-to-post
```

---

### Hugging Face Spaces Deployment

1. **Create Space**
- Go to huggingface.co/spaces
- Click "Create new Space"
- Choose SDK: Docker
- Name: voice-to-post

2. **Upload Files**
```bash
git remote add hf https://huggingface.co/spaces/YourUsername/voice-to-post
git push hf main
```

3. **Configure Secrets**
In Space Settings → Repository Secrets, add all environment variables

4. **Create Dataset for DB**
- Create dataset: `YourUsername/voice-to-post-data`
- Get write token
- Add as `HF_TOKEN` secret

---

## 🔑 Environment Variables

### Required

```bash
# Encryption (CRITICAL - Generate first!)
ENCRYPTION_KEY="<fernet-key>"

# Speech-to-Text
DEEPGRAM_API_KEY="<your-deepgram-key>"

# LLM
GEMINI_API_KEY="<your-google-ai-key>"

# Database Persistence
HF_TOKEN="<your-huggingface-write-token>"

# OAuth - Twitter
TWITTER_CLIENT_ID="<your-twitter-client-id>"
TWITTER_CLIENT_SECRET="<your-twitter-client-secret>"

# OAuth - LinkedIn
LINKEDIN_CLIENT_ID="<your-linkedin-client-id>"
LINKEDIN_CLIENT_SECRET="<your-linkedin-client-secret>"

# OAuth - Discord
DISCORD_CLIENT_ID="<your-discord-client-id>"
DISCORD_CLIENT_SECRET="<your-discord-client-secret>"

# Base URL (for OAuth callbacks)
BASE_URL="https://your-space.hf.space"  # or http://localhost:7860
```

### Optional

```bash
# News Enrichment (optional)
NEWS_API_KEY="<your-newsapi-key>"
```

### Generate Encryption Key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## 📊 Performance Metrics

### Typical Request Times
- **Transcription**: 2-5 seconds
- **Vector Search**: <100ms
- **LLM Generation**: 10-15 seconds
- **Scoring**: <50ms
- **Publishing**: 1-3 seconds
- **Total End-to-End**: 15-20 seconds

### Timeout Configuration
- **Speech Service**: 60 seconds
- **Generate Post**: 35 seconds (LLM only)
- **Overall Request**: 20 seconds (transcription)
- **SQLite Operations**: 30 seconds

### Resource Usage
- **Memory**: ~2GB (mainly from SentenceTransformer model)
- **Disk**: Minimal (SQLite DB <10MB)
- **CPU**: Moderate during transcription/generation

---

## 🔧 Maintenance & Operations

### Database Backup
- Automatic: Every credential change uploads to HF Hub
- Manual: Copy `/tmp/credentials.db` file
- Restore: Place file in HF dataset repo

### Model Updates
- Vector model cached locally after first load
- LLM version controlled via `langchain-google-genai`
- Update by changing model name in `generation_service.py`

### Monitoring
- Check logs for API errors
- Monitor HF Hub sync status
- Track OAuth token expiration (Twitter refresh tokens)

### Debugging
- Enable debug logging: Set `DEBUG=true`
- Check raw LLM output in generation service logs
- Verify vector store indexing with search tests

---

## 🎓 Key Design Decisions

### Why FAISS?
- Fast similarity search (Facebook AI)
- Works in-memory (no external DB)
- Scales to millions of vectors
- Perfect for user-scoped RAG

### Why Gemini 2.5 Flash?
- Fast response times (<15s)
- Strong instruction following
- JSON output reliability
- Cost-effective at scale

### Why WAL Mode?
- Cloud-friendly (HF Spaces)
- Concurrent read/write
- No "database locked" errors

### Why Single Generation Call?
- Original design: retry loop caused timeouts (150s potential)
- New design: one call, score all, sort
- Result: 10x faster, no timeouts

### Why User-Scoped Vector Store?
- Privacy: users only retrieve their own context
- Relevance: better personalization
- Scalability: filter at search time

---

## 📝 Notes for Team

1. **The scoring system is tuned for variety** - posts will have different scores, helping users choose
2. **The prompt is strict** - minimal hallucination with anti-fabrication rules
3. **OAuth tokens are encrypted** - never log or expose encrypted values
4. **Vector store is in-memory** - resets on restart (but user data persists via DB sync)
5. **Platform character limits** - enforced in prompt, not code
6. **Deep links** - mobile apps should handle `yourapp://callback` scheme

---

## 🐛 Known Issues & Limitations

1. **Vector store resets on restart** - past posts need re-indexing
2. **No user authentication** - relies on `user_id` parameter (honor system)
3. **Wide-open CORS** - should be restricted in production
4. **Twitter refresh tokens** - not implemented (would need token refresh endpoint)
5. **No rate limiting** - vulnerable to abuse
6. **Single-language** - English only (both transcription and generation)

---

**Last Updated**: 2026-08-23  
**Version**: 1.0  
**Maintainer**: Voice-To-Post Team
