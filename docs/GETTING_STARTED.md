# 🚀 Getting Started - Voice-To-Post

**Complete setup in 10 minutes**

---

## Step 1: Install Dependencies (2 minutes)

```bash
cd Voice-To-Post
pip install -r requirements.txt
```

---

## Step 2: Get API Keys (5 minutes)

### Required (You Already Have)
- ✅ DEEPGRAM_API_KEY
- ✅ GEMINI_API_KEY  
- ✅ HF_TOKEN
- ✅ ENCRYPTION_KEY

### Optional (For Images)
**Pexels** (2 min): https://www.pexels.com/api/
**Unsplash** (2 min): https://unsplash.com/developers

---

## Step 3: Configure Environment (1 minute)

Create `.env` file:

```bash
# Required
DEEPGRAM_API_KEY=your-existing-key
GEMINI_API_KEY=your-existing-key
HF_TOKEN=your-existing-token
ENCRYPTION_KEY=your-existing-key

# OAuth (if using social publishing)
TWITTER_CLIENT_ID=your-id
TWITTER_CLIENT_SECRET=your-secret
LINKEDIN_CLIENT_ID=your-id
LINKEDIN_CLIENT_SECRET=your-secret
DISCORD_CLIENT_ID=your-id
DISCORD_CLIENT_SECRET=your-secret

# Optional (for images)
PEXELS_API_KEY=your-new-key
UNSPLASH_ACCESS_KEY=your-new-key

# Settings
BASE_URL=http://localhost:7860
ENVIRONMENT=development
```

---

## Step 4: Run (1 minute)

```bash
# Use production version (recommended)
cp main_production.py main.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## Step 5: Test (1 minute)

```bash
# Health check
curl http://localhost:7860/

# System info
curl http://localhost:7860/system/info
```

**Expected response:**
```json
{
  "status": "Voice-To-Post Backend v2.0 is running",
  "version": "2.0.0",
  "features": [...]
}
```

---

## ✅ You're Ready!

### Next Steps:

**For Frontend Team:**
- Read `docs/FRONTEND_INTEGRATION_GUIDE.md`
- Start integrating APIs

**For Research Paper:**
- Read `docs/RESEARCH_PAPER_SUMMARY.md`
- Run your benchmarks

**For Testing:**
- Test endpoints with Postman or curl
- Check `docs/` for examples

---

## 🎯 Quick Test

Generate a post:

```bash
# Create test audio file
echo "Test productivity tips" > test.txt

# Or use actual audio file
curl -X POST http://localhost:7860/generate-post \
  -F "audio_file=@your_audio.wav" \
  -F "tone=professional" \
  -F "platform=twitter" \
  -F "user_id=test123"
```

---

## 📚 Documentation

All docs in `docs/` folder:
- `FRONTEND_INTEGRATION_GUIDE.md` - API examples
- `RESEARCH_PAPER_SUMMARY.md` - Paper content
- `PROJECT_DOCUMENTATION.md` - Technical details

---

## 🆘 Issues?

**Dependencies fail?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port already in use?**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Images not working?**
- Check API keys in `.env`
- Images are optional - other features work without them

---

**Total Setup Time: ~10 minutes** ⚡

**Happy building!** 🚀
