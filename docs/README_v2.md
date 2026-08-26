# Voice-To-Post v2.0 - Enhanced Backend 🚀

## 🎉 What's New in v2.0

### ✅ All Features Using FREE Services Only!

1. **🎨 AI Image Generation** (FREE)
   - Pexels API integration (200 requests/hour)
   - Unsplash API integration (50 requests/hour)
   - Hugging Face Stable Diffusion (FREE with rate limits)
   - Multiple image options per post
   - Platform-optimized sizing

2. **🧵 Thread Generator** (FREE)
   - Multi-post threads from single voice recording
   - Auto-numbered posts (1/5, 2/5, etc.)
   - Platform-specific character limits
   - Cohesive narrative flow

3. **✏️ Post Refinement** (FREE)
   - 13 refinement types (shorten, lengthen, change tone, etc.)
   - AI-powered post editor
   - Quality analysis with suggestions
   - Batch refinement support

4. **#️⃣ Smart Hashtag Suggestions** (FREE)
   - AI-generated relevant hashtags
   - Platform-appropriate recommendations
   - Trend-aware suggestions

5. **🌍 Cross-Platform Generator** (FREE)
   - Generate optimized posts for multiple platforms simultaneously
   - Single voice → Twitter + LinkedIn + Discord posts

6. **🔐 JWT Authentication** (FREE)
   - Secure user accounts
   - API key support for programmatic access
   - Password hashing with bcrypt

7. **⚡ Rate Limiting** (FREE)
   - Prevents API abuse
   - Per-endpoint limits
   - Per-user limits

8. **💾 Vector Store Persistence** (FREE)
   - FAISS index saves to disk
   - Syncs to Hugging Face Hub
   - No data loss on restart

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file:

```bash
# CRITICAL: Generate encryption key first
ENCRYPTION_KEY="<your-fernet-key>"

# Core APIs (Required)
DEEPGRAM_API_KEY="<your-deepgram-key>"
GEMINI_API_KEY="<your-google-ai-key>"
HF_TOKEN="<your-huggingface-token>"

# OAuth (Required for social posting)
TWITTER_CLIENT_ID="<your-twitter-client-id>"
TWITTER_CLIENT_SECRET="<your-twitter-client-secret>"
LINKEDIN_CLIENT_ID="<your-linkedin-client-id>"
LINKEDIN_CLIENT_SECRET="<your-linkedin-client-secret>"
DISCORD_CLIENT_ID="<your-discord-client-id>"
DISCORD_CLIENT_SECRET="<your-discord-client-secret>"

# Image Generation (FREE - Optional but recommended)
PEXELS_API_KEY="<your-pexels-key>"  # Get FREE at https://www.pexels.com/api/
UNSPLASH_ACCESS_KEY="<your-unsplash-key>"  # Get FREE at https://unsplash.com/developers

# Authentication (Optional - generates random if not set)
JWT_SECRET_KEY="<your-jwt-secret>"  # For secure authentication

# Base URL
BASE_URL="https://your-space.hf.space"  # or http://localhost:7860

# Optional
NEWS_API_KEY="<your-newsapi-key>"  # Optional news enrichment
```

### 3. Generate Encryption Key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 4. Get FREE API Keys

#### Pexels (FREE)
1. Go to https://www.pexels.com/api/
2. Sign up (free)
3. Get API key instantly
4. **Rate Limit**: 200 requests/hour

#### Unsplash (FREE)
1. Go to https://unsplash.com/developers
2. Create an app (free)
3. Get Access Key
4. **Rate Limit**: 50 requests/hour

## 🚀 Running the Application

### Option 1: Use Enhanced Version

```bash
# Rename enhanced main to main
mv main_enhanced.py main.py

# Run
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### Option 2: Keep Both Versions

```bash
# Run enhanced version on different port
uvicorn main_enhanced:app --host 0.0.0.0 --port 7861
```

## 🆕 New API Endpoints

### Authentication

#### Register User
```http
POST /auth/register
Content-Type: multipart/form-data

email: user@example.com
password: securepassword123
full_name: John Doe (optional)
```

**Response:**
```json
{
  "status": "success",
  "user_id": "abc123def456",
  "email": "user@example.com",
  "access_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

#### Login
```http
POST /auth/login
Content-Type: multipart/form-data

email: user@example.com
password: securepassword123
```

#### Create API Key
```http
POST /auth/create-api-key
Authorization: Bearer <your-jwt-token>
Content-Type: multipart/form-data

name: "My Mobile App" (optional)
```

#### Get Current User
```http
GET /auth/me
Authorization: Bearer <your-jwt-token>
```

---

### Image Generation

#### Generate Post WITH Images
```http
POST /generate-post-with-image
Content-Type: multipart/form-data

audio_file: <file>
tone: professional
platform: twitter
user_id: abc123
image_method: stock  # or "ai"
num_image_options: 3
```

**Response:**
```json
{
  "status": "success",
  "transcript": "Tips for remote work...",
  "variations": [...],
  "images": [
    {
      "image_base64": "iVBORw0KG...",
      "source": "pexels",
      "photographer": "Jane Doe",
      "keywords": ["business", "remote", "work"]
    }
  ],
  "image_count": 3
}
```

#### Generate Image for Existing Post
```http
POST /generate-image-for-post
Content-Type: multipart/form-data

post_text: "Your post content here"
platform: twitter
method: stock  # or "ai"
num_options: 3
```

---

### Thread Generation

#### Generate Thread
```http
POST /generate-thread
Content-Type: multipart/form-data

audio_file: <file>
platform: twitter
tone: professional
user_id: abc123
max_posts: 5
```

**Response:**
```json
{
  "status": "success",
  "transcript": "Long content...",
  "thread": [
    {
      "post_number": 1,
      "text": "First post in thread...\n\n(1/5)"
    },
    {
      "post_number": 2,
      "text": "Second post...\n\n(2/5)"
    }
  ],
  "total_posts": 5
}
```

---

### Cross-Platform Generation

#### Generate for Multiple Platforms
```http
POST /generate-cross-platform
Content-Type: multipart/form-data

audio_file: <file>
platforms: "twitter,linkedin,discord"  # Comma-separated
tone: professional
user_id: abc123
```

**Response:**
```json
{
  "status": "success",
  "transcript": "Your content...",
  "posts": {
    "twitter": {
      "text": "Short punchy tweet...",
      "platform": "twitter"
    },
    "linkedin": {
      "text": "Detailed LinkedIn post...",
      "platform": "linkedin"
    },
    "discord": {
      "text": "Discord announcement...",
      "platform": "discord"
    }
  }
}
```

---

### Post Refinement

#### Refine Post
```http
POST /refine-post
Content-Type: multipart/form-data

post_text: "Original post here"
refinement_type: shorten  # or lengthen, more_formal, add_humor, etc.
platform: twitter
custom_instruction: "Make it funnier" (optional)
```

**Response:**
```json
{
  "status": "success",
  "original": "Original post here...",
  "refined": "Refined shorter version...",
  "refinement_type": "shorten"
}
```

#### Get Available Refinement Types
```http
GET /refinement-types
```

**Response:**
```json
{
  "refinement_types": {
    "shorten": "Reduce word count while keeping the message",
    "lengthen": "Add more detail and context",
    "more_formal": "Professional business tone",
    "add_humor": "Make it funny and entertaining",
    ...
  }
}
```

#### Analyze Post Quality
```http
POST /analyze-post
Content-Type: multipart/form-data

post_text: "Your post here"
platform: twitter
```

**Response:**
```json
{
  "status": "success",
  "analysis": {
    "readability_score": 8,
    "engagement_potential": 7,
    "clarity_score": 9,
    "tone_appropriateness": 8,
    "strengths": ["Clear message", "Good hook"],
    "weaknesses": ["Could use hashtags"],
    "suggestions": [
      "Add 2-3 relevant hashtags",
      "Consider adding an emoji",
      "End with a call-to-action"
    ]
  }
}
```

---

### Smart Hashtags

#### Get Hashtag Suggestions
```http
POST /suggest-hashtags
Content-Type: multipart/form-data

post_text: "Your post content"
platform: twitter
num_hashtags: 5
```

**Response:**
```json
{
  "status": "success",
  "hashtags": ["SocialMedia", "ContentCreation", "DigitalMarketing"],
  "formatted": ["#SocialMedia", "#ContentCreation", "#DigitalMarketing"]
}
```

---

### System Info

#### Get System Information
```http
GET /system/info
```

**Response:**
```json
{
  "version": "2.0.0",
  "features": {
    "voice_to_post": true,
    "image_generation": {
      "available": true,
      "methods": ["stock", "ai"]
    },
    "thread_generation": true,
    "post_refinement": true,
    "smart_hashtags": true,
    "multi_platform": true,
    "authentication": true,
    "rate_limiting": true,
    "vector_persistence": true
  },
  "rate_limits": {
    "auth": "10/minute",
    "generation": "5/minute",
    "publish": "20/minute"
  }
}
```

#### Get Vector Store Stats
```http
GET /vector-store/stats
```

## 📊 Rate Limits

| Endpoint Type | Limit | Purpose |
|--------------|-------|---------|
| Authentication | 10/min | Login, register |
| Generation | 5/min | Post/image generation |
| Publishing | 20/min | Social media posting |
| Upload | 10/min | File uploads |
| General | 60/min | Other endpoints |
| Analytics | 30/min | Analytics queries |

## 🔧 Migration from v1.0 to v2.0

### Database Migration
The new version adds authentication tables automatically. Your existing data remains intact.

```bash
# Backup your database first (optional)
cp /tmp/credentials.db /tmp/credentials.db.backup

# Run the enhanced version - tables auto-create
uvicorn main_enhanced:app --host 0.0.0.0 --port 7860
```

### Backward Compatibility
All v1.0 endpoints still work! The enhanced version is 100% backward compatible.

## 🎯 Feature Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Voice-to-Post | ✅ | ✅ |
| Multi-Platform | ✅ | ✅ |
| RAG Context | ✅ | ✅ |
| Image Generation | ❌ | ✅ FREE |
| Thread Generator | ❌ | ✅ |
| Post Refinement | ❌ | ✅ |
| Smart Hashtags | ❌ | ✅ |
| Authentication | ❌ | ✅ |
| Rate Limiting | ❌ | ✅ |
| Vector Persistence | ❌ | ✅ |
| Cross-Platform Gen | ❌ | ✅ |

## 💰 Cost Analysis

### v2.0 Total Cost: $0/month

All new features use FREE services:

| Service | Cost | Usage |
|---------|------|-------|
| Pexels API | FREE | 200 requests/hour |
| Unsplash API | FREE | 50 requests/hour |
| HF Stable Diffusion | FREE | Rate-limited |
| Gemini API | Your existing key | - |
| Deepgram | Your existing key | - |
| JWT/bcrypt | Library (free) | - |
| slowapi | Library (free) | - |
| FAISS persistence | Library (free) | - |

**No additional costs!** 🎉

## 🐛 Troubleshooting

### Image Generation Fails
```python
# Check API keys
print(os.getenv("PEXELS_API_KEY"))
print(os.getenv("UNSPLASH_ACCESS_KEY"))

# Fallback to HF if stock APIs not available
# Hugging Face model might be loading (first request takes longer)
```

### Rate Limit Hit
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please slow down.",
  "retry_after": "60 seconds"
}
```
Wait 60 seconds and retry.

### Authentication Issues
```bash
# Generate new JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
JWT_SECRET_KEY="<new-secret>"
```

## 📚 Documentation Files

- `PROJECT_DOCUMENTATION.md` - Complete technical documentation
- `FEATURE_ENHANCEMENTS.md` - All feature ideas and roadmap
- `FREE_FEATURES.md` - Summary of free services
- `README.md` - This file
- `RESEARCH_PAPER_SUMMARY.md` - Research paper content

## 🎓 For Research Paper

See `RESEARCH_PAPER_SUMMARY.md` for:
- Problem statement
- Solution architecture
- Novel contributions
- Implementation details
- Results and evaluation
- Future work

## 🤝 Contributing

This is a research project. All features are implemented using free, open-source solutions.

## 📄 License

MIT License - Free for research and educational purposes

## 🙏 Credits

- **Deepgram** - Speech-to-text
- **Google Gemini** - LLM
- **Hugging Face** - Model hosting & Stable Diffusion
- **Pexels** - Free stock photos
- **Unsplash** - Free stock photos
- **FastAPI** - Web framework
- **FAISS** - Vector search

---

**Voice-To-Post v2.0** - Empowering creators with AI 🚀
