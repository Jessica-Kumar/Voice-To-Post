# 🚀 Deploying to Hugging Face Spaces - Voice-To-Post v2.0

## Your Current Setup

✅ You already have Voice-To-Post v1.0 running on Hugging Face Spaces  
🆕 Now deploying v2.0 with all new features

---

## 📋 Deployment Checklist

### Step 1: Update Files on Hugging Face (5 minutes)

**Option A: Via Git (Recommended)**

```bash
# Clone your existing HF Space
git clone https://huggingface.co/spaces/YourUsername/voice-to-post
cd voice-to-post

# Copy new files from your local project
cp /path/to/Voice-To-Post/main_production.py main.py
cp /path/to/Voice-To-Post/requirements.txt .
cp /path/to/Voice-To-Post/image_service.py .
cp /path/to/Voice-To-Post/auth_service.py .
cp /path/to/Voice-To-Post/rate_limiter.py .
cp /path/to/Voice-To-Post/thread_service.py .
cp /path/to/Voice-To-Post/refinement_service.py .
cp /path/to/Voice-To-Post/vector_store.py .

# Commit and push
git add .
git commit -m "Update to v2.0 - Add images, threads, refinement, auth"
git push
```

**Option B: Via Web Interface**

1. Go to https://huggingface.co/spaces/YourUsername/voice-to-post
2. Click "Files" tab
3. Upload new files one by one:
   - `main.py` (use `main_production.py`)
   - `requirements.txt`
   - `image_service.py`
   - `auth_service.py`
   - `rate_limiter.py`
   - `thread_service.py`
   - `refinement_service.py`
   - `vector_store.py` (updated)

---

### Step 2: Add New Environment Variables (3 minutes)

Go to **Settings → Repository secrets** and add:

**New Variables (Optional - for images):**
```bash
PEXELS_API_KEY=your-key-from-pexels
UNSPLASH_ACCESS_KEY=your-key-from-unsplash
```

**Optional Settings:**
```bash
ENVIRONMENT=production
FRONTEND_URLS=https://your-frontend.com
```

**Your existing variables stay the same:**
- ✅ DEEPGRAM_API_KEY
- ✅ GEMINI_API_KEY
- ✅ HF_TOKEN
- ✅ ENCRYPTION_KEY
- ✅ TWITTER_CLIENT_ID, TWITTER_CLIENT_SECRET
- ✅ LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET
- ✅ DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET
- ✅ BASE_URL (update to your Space URL)

---

### Step 3: Verify Deployment (2 minutes)

Once Hugging Face rebuilds (takes ~3-5 minutes):

**Test Health:**
```bash
curl https://your-username-voice-to-post.hf.space/
```

**Test System Info:**
```bash
curl https://your-username-voice-to-post.hf.space/system/info
```

**Expected Response:**
```json
{
  "status": "Voice-To-Post Backend v2.0 is running",
  "version": "2.0.0",
  "features": [
    "Voice-to-Post Generation",
    "AI Image Generation (FREE)",
    "Thread Generator",
    ...
  ]
}
```

---

## 🔄 Migration Path

### What Changes:
- ✅ 6 new service files added
- ✅ `main.py` updated to v2.0
- ✅ `requirements.txt` updated (4 new libraries)
- ✅ `vector_store.py` updated (persistence)

### What Stays the Same:
- ✅ All v1.0 endpoints still work (backward compatible)
- ✅ Existing users won't break
- ✅ Database structure compatible
- ✅ OAuth flows unchanged

### What's New:
- 🆕 26 total endpoints (was 12)
- 🆕 Image generation
- 🆕 Thread generation
- 🆕 Post refinement
- 🆕 Smart hashtags
- 🆕 Authentication
- 🆕 Rate limiting

---

## 📝 Update Your Space README

Update the README on your Hugging Face Space:

```markdown
---
title: Voice To Post v2.0
emoji: 🦀
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
---

# Voice-To-Post v2.0

AI-powered voice-to-social-media platform with RAG, multi-platform support, and FREE image generation.

## What's New in v2.0
- 🎨 AI Image Generation (FREE)
- 🧵 Thread Generator
- ✏️ Post Refinement (13 types)
- #️⃣ Smart Hashtags
- 🔐 Authentication
- ⚡ Rate Limiting
- 💾 Vector Persistence

## Features
- 🎤 Voice-to-post generation
- 🌐 4 platforms (Twitter, LinkedIn, Discord, Medium)
- 🤖 RAG-based personalization
- 📊 Quality scoring (0.847 avg)
- ⚡ 15-20s latency

## API Documentation
See `/system/info` for available endpoints.

## Cost
$0/month - All free APIs!
```

---

## 🐛 Troubleshooting

### Build Fails?

**Check Dockerfile:**
Make sure your `Dockerfile` is still correct:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Space Crashes on Start?

**Check Logs:**
1. Go to your Space
2. Click "Logs" tab
3. Look for error messages

**Common Issues:**
- ❌ Missing environment variable → Add it in Settings
- ❌ Import error → Check `requirements.txt` uploaded correctly
- ❌ Port conflict → Make sure Dockerfile uses port 7860

### Images Not Working?

**Check API Keys:**
```bash
# In Space logs, you should NOT see errors about:
# - "PEXELS_API_KEY not set"
# - "UNSPLASH_ACCESS_KEY not set"
```

If you see these but don't want images yet, that's OK! All other features work.

---

## 🚀 Deployment Steps Summary

### Quick Deployment (10 minutes)

```bash
# 1. Clone your HF Space
git clone https://huggingface.co/spaces/YourUsername/voice-to-post
cd voice-to-post

# 2. Copy all new files
cp ../Voice-To-Post/main_production.py main.py
cp ../Voice-To-Post/*.py .
cp ../Voice-To-Post/requirements.txt .

# 3. Commit and push
git add .
git commit -m "🚀 Update to v2.0 - All new features"
git push

# 4. Add secrets in HF web interface (if needed)
# - PEXELS_API_KEY (optional)
# - UNSPLASH_ACCESS_KEY (optional)

# 5. Wait for rebuild (~3-5 min)

# 6. Test
curl https://your-space.hf.space/system/info
```

**Done!** 🎉

---

## 📊 What Your Users Will See

### Before (v1.0):
- Voice-to-post generation
- Multi-platform publishing
- RAG personalization

### After (v2.0):
- ✅ Everything from v1.0 (still works!)
- 🆕 Post comes with image options
- 🆕 Can create threads
- 🆕 Can refine posts
- 🆕 Get hashtag suggestions
- 🆕 Generate for all platforms at once

**No breaking changes!** Old API calls still work.

---

## 🎯 Post-Deployment Verification

### Test Core Features:
```bash
BASE_URL="https://your-username-voice-to-post.hf.space"

# 1. Health check
curl $BASE_URL/

# 2. System info
curl $BASE_URL/system/info

# 3. Vector stats
curl $BASE_URL/vector-store/stats

# 4. Refinement types
curl $BASE_URL/refinement-types
```

### Test New Endpoints:
```bash
# Test image generation
curl -X POST $BASE_URL/generate-image-for-post \
  -F "post_text=AI is amazing" \
  -F "platform=twitter"

# Test hashtag suggestions
curl -X POST $BASE_URL/suggest-hashtags \
  -F "post_text=Just launched my AI app" \
  -F "platform=twitter"
```

---

## 📱 Update Your Frontend

Once deployed, update your frontend to point to new Space URL:

```javascript
// Old
const API_URL = "https://your-old-space.hf.space";

// New (same URL, just updated backend)
const API_URL = "https://your-username-voice-to-post.hf.space";

// All v1.0 endpoints still work!
// Plus 14 new endpoints available
```

---

## 🎉 You're Live!

Once deployed:
- ✅ v2.0 running on Hugging Face
- ✅ All new features available
- ✅ Backward compatible
- ✅ Free hosting
- ✅ Auto-scaling
- ✅ HTTPS included

**Share the URL with your team!**

---

## 📞 Need Help?

**Space not building?**
- Check Dockerfile is correct
- Check requirements.txt uploaded
- Check logs in HF interface

**Features not working?**
- Check environment variables in Settings
- Check logs for import errors
- Test locally first with same .env

**Old endpoints broken?**
- Shouldn't happen (backward compatible)
- Check you uploaded `main_production.py` as `main.py`
- Check database.py and other core files uploaded

---

## 🔗 Resources

**Your HF Space:** https://huggingface.co/spaces/YourUsername/voice-to-post  
**HF Docs:** https://huggingface.co/docs/hub/spaces  
**API Docs:** See `docs/FRONTEND_INTEGRATION_GUIDE.md`

---

**Deployment Time:** ~10 minutes  
**Downtime:** ~3-5 minutes (during rebuild)  
**Breaking Changes:** None (backward compatible)  

🚀 **Ready to deploy!**
