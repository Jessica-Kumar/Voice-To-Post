# 🎯 FRONTEND INTEGRATION GUIDE - Voice-To-Post v2.0

## 📋 Quick Answer: YES, Backend is Production-Ready!

**Status**: ✅ **PRODUCTION READY** with minor recommendations below

**What Works Perfectly**:
- ✅ All core endpoints functional
- ✅ Backward compatible with v1.0
- ✅ Comprehensive error handling
- ✅ Rate limiting active
- ✅ Authentication optional (not breaking)
- ✅ CORS enabled
- ✅ All new features tested

**Minor Improvements Recommended** (not blocking):
- File size limits (10MB max recommended)
- Environment-based CORS (production vs dev)
- Optional: Image URL instead of base64

---

## 🚀 Getting Started (Frontend Team)

### 1. Backend URL
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:7860";
```

### 2. Health Check
```javascript
// Check if backend is ready
async function checkBackendHealth() {
    const response = await fetch(`${API_BASE_URL}/`);
    const data = await response.json();
    console.log("Backend status:", data.status);
    console.log("Available features:", data.features);
    return data;
}
```

### 3. Error Handling Pattern
```javascript
async function callAPI(endpoint, formData) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.message || "API Error");
        }
        
        return data;
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
}
```

---

## 📡 Core API Endpoints (Frontend Examples)

### 1. Generate Post from Voice

**Endpoint**: `POST /generate-post`

**JavaScript/React Example**:
```javascript
async function generatePost(audioFile, tone, platform, userId) {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('tone', tone);
    formData.append('platform', platform);
    formData.append('user_id', userId);
    
    try {
        const response = await fetch(`${API_BASE_URL}/generate-post`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === "success") {
            // data.variations = array of 5 posts sorted by score
            // data.transcript = what user said
            return data;
        } else {
            throw new Error(data.message || "Generation failed");
        }
    } catch (error) {
        if (error.message.includes("timeout")) {
            alert("Audio too long or network slow. Try shorter recording.");
        } else {
            alert("Error: " + error.message);
        }
        throw error;
    }
}

// Usage
const audioBlob = await recordAudio(); // Your recording logic
const result = await generatePost(audioBlob, "professional", "twitter", "user123");

console.log("Transcript:", result.transcript);
console.log("Best post:", result.variations[0].text);
console.log("Score:", result.variations[0].score);
```

**Response Format**:
```json
{
  "status": "success",
  "transcript": "Here are some productivity tips...",
  "variations": [
    {
      "text": "🚀 Boost your productivity with these 3 tips:\n\n1. Time blocking\n2. Single-tasking\n3. Regular breaks\n\n#Productivity #WorkSmart",
      "score": 0.892,
      "breakdown": {
        "ai_confidence": 0.9,
        "retrieval_relevance": 0.85,
        "safety_score": 1.0,
        "engagement_potential": 0.75
      }
    },
    // ... 4 more variations
  ],
  "total_generated": 5,
  "attempts_used": 1
}
```

---

### 2. Generate Post WITH Images

**Endpoint**: `POST /generate-post-with-image`

**JavaScript Example**:
```javascript
async function generatePostWithImage(audioFile, tone, platform, userId, imageMethod = "stock") {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('tone', tone);
    formData.append('platform', platform);
    formData.append('user_id', userId);
    formData.append('image_method', imageMethod); // "stock" or "ai"
    formData.append('num_image_options', 3);
    
    const response = await fetch(`${API_BASE_URL}/generate-post-with-image`, {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    return data;
}

// Usage
const result = await generatePostWithImage(audioBlob, "casual", "twitter", "user123", "stock");

console.log("Post:", result.variations[0].text);
console.log("Images:", result.images); // Array of 3 image options

// Display images
result.images.forEach((img, index) => {
    const imgElement = document.createElement('img');
    imgElement.src = `data:image/jpeg;base64,${img.image_base64}`;
    imgElement.alt = `Option ${index + 1}`;
    document.body.appendChild(imgElement);
});
```

**Response Format**:
```json
{
  "status": "success",
  "transcript": "...",
  "variations": [/* posts */],
  "images": [
    {
      "image_base64": "iVBORw0KGgoAAAANSUhE...", // Base64 encoded
      "thumbnail_url": "https://images.pexels.com/...",
      "source": "pexels",
      "photographer": "John Doe",
      "keywords": ["business", "productivity", "office"]
    },
    // 2 more image options
  ],
  "image_count": 3
}
```

⚠️ **Important**: Images are base64-encoded in response. Each image is ~500KB-2MB. Consider:
```javascript
// Option 1: Display immediately
img.src = `data:image/jpeg;base64,${imageData.image_base64}`;

// Option 2: Convert to blob for upload
function base64ToBlob(base64) {
    const byteString = atob(base64);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: 'image/jpeg' });
}

const blob = base64ToBlob(imageData.image_base64);
```

---

### 3. Generate Thread (Multi-Post)

**Endpoint**: `POST /generate-thread`

**JavaScript Example**:
```javascript
async function generateThread(audioFile, platform, tone, userId, maxPosts = 5) {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('platform', platform);
    formData.append('tone', tone);
    formData.append('user_id', userId);
    formData.append('max_posts', maxPosts);
    
    const response = await fetch(`${API_BASE_URL}/generate-thread`, {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Usage
const thread = await generateThread(audioBlob, "twitter", "professional", "user123", 5);

console.log("Thread posts:");
thread.thread.forEach(post => {
    console.log(`Post ${post.post_number}:`, post.text);
});
```

**Response**:
```json
{
  "status": "success",
  "transcript": "Long content about remote work...",
  "thread": [
    {
      "post_number": 1,
      "text": "🏠 Remote work is transforming how we collaborate...\n\n(1/5)"
    },
    {
      "post_number": 2,
      "text": "Key benefits include flexibility and work-life balance...\n\n(2/5)"
    },
    // ... 3 more posts
  ],
  "total_posts": 5,
  "platform": "twitter"
}
```

---

### 4. Refine Existing Post

**Endpoint**: `POST /refine-post`

**JavaScript Example**:
```javascript
async function refinePost(postText, refinementType, platform = "twitter") {
    const formData = new FormData();
    formData.append('post_text', postText);
    formData.append('refinement_type', refinementType);
    formData.append('platform', platform);
    
    const response = await fetch(`${API_BASE_URL}/refine-post`, {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Available refinement types
const REFINEMENT_TYPES = [
    "shorten",
    "lengthen",
    "more_formal",
    "more_casual",
    "add_humor",
    "add_hooks",
    "add_cta",
    "remove_jargon",
    "add_emojis",
    "remove_emojis",
    "add_hashtags",
    "more_professional",
    "more_engaging"
];

// Usage
const original = "This is my post about productivity tips";
const refined = await refinePost(original, "add_humor", "twitter");

console.log("Original:", refined.original);
console.log("Refined:", refined.refined);
```

**Response**:
```json
{
  "status": "success",
  "original": "This is my post about productivity tips",
  "refined": "Want to know the secret to productivity? 🤔\n\nSpoiler: It's not another app! Here are my battle-tested tips that actually work... 💪\n\n#ProductivityHacks #WorkSmart",
  "refinement_type": "add_humor"
}
```

---

### 5. Get Smart Hashtag Suggestions

**Endpoint**: `POST /suggest-hashtags`

**JavaScript Example**:
```javascript
async function suggestHashtags(postText, platform = "twitter", numHashtags = 5) {
    const formData = new FormData();
    formData.append('post_text', postText);
    formData.append('platform', platform);
    formData.append('num_hashtags', numHashtags);
    
    const response = await fetch(`${API_BASE_URL}/suggest-hashtags`, {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Usage
const hashtags = await suggestHashtags(
    "Just launched my AI-powered productivity app!",
    "twitter",
    5
);

console.log("Suggested hashtags:", hashtags.formatted);
// ["#AITools", "#ProductivityApp", "#TechLaunch", "#Startup", "#Innovation"]
```

---

### 6. Publish to Social Media

**Endpoint**: `POST /publish-post`

**JavaScript Example**:
```javascript
async function publishPost(platform, postText, userId) {
    const formData = new FormData();
    formData.append('platform', platform);
    formData.append('post_text', postText);
    formData.append('user_id', userId);
    
    const response = await fetch(`${API_BASE_URL}/publish-post`, {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    if (data.status === "success") {
        return data;
    } else {
        throw new Error(data.message || "Publishing failed");
    }
}

// Usage
try {
    const result = await publishPost("twitter", postText, "user123");
    console.log("Published!", result.url);
    alert(`Successfully posted to ${result.platform}! URL: ${result.url}`);
} catch (error) {
    if (error.message.includes("credentials")) {
        alert("Please connect your Twitter account first!");
    } else {
        alert("Publishing failed: " + error.message);
    }
}
```

**Success Response**:
```json
{
  "status": "success",
  "platform": "twitter",
  "post_id": "1234567890",
  "url": "https://twitter.com/user/status/1234567890",
  "message": "Successfully posted to Twitter!"
}
```

**Error Response**:
```json
{
  "status": "error",
  "message": "No credentials found for user user123."
}
```

---

### 7. Cross-Platform Generation

**Endpoint**: `POST /generate-cross-platform`

**JavaScript Example**:
```javascript
async function generateCrossPlatform(audioFile, platforms, tone, userId) {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('platforms', platforms.join(',')); // "twitter,linkedin,discord"
    formData.append('tone', tone);
    formData.append('user_id', userId);
    
    const response = await fetch(`${API_BASE_URL}/generate-cross-platform`, {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Usage
const posts = await generateCrossPlatform(
    audioBlob,
    ['twitter', 'linkedin', 'discord'],
    'professional',
    'user123'
);

console.log("Twitter post:", posts.posts.twitter.text);
console.log("LinkedIn post:", posts.posts.linkedin.text);
console.log("Discord post:", posts.posts.discord.text);
```

**Response**:
```json
{
  "status": "success",
  "transcript": "...",
  "posts": {
    "twitter": {
      "text": "Short, punchy tweet with emojis 🚀 #AI #Tech",
      "platform": "twitter"
    },
    "linkedin": {
      "text": "Detailed professional post with context...\n\nKey insights:\n- Point 1\n- Point 2\n\n#ProfessionalDevelopment #Leadership",
      "platform": "linkedin"
    },
    "discord": {
      "text": "Hey @everyone! Exciting announcement...",
      "platform": "discord"
    }
  },
  "platforms": ["twitter", "linkedin", "discord"]
}
```

---

## 🔐 Authentication (Optional)

### Register User
```javascript
async function registerUser(email, password, fullName) {
    const formData = new FormData();
    formData.append('email', email);
    formData.append('password', password);
    formData.append('full_name', fullName);
    
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    // Save token
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('user_id', data.user_id);
    
    return data;
}
```

### Login
```javascript
async function login(email, password) {
    const formData = new FormData();
    formData.append('email', email);
    formData.append('password', password);
    
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('user_id', data.user_id);
    
    return data;
}
```

### Use Token in Requests (If using auth)
```javascript
async function callAuthenticatedAPI(endpoint, formData) {
    const token = localStorage.getItem('access_token');
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: formData
    });
    
    return await response.json();
}
```

---

## ⚠️ Error Handling Guide

### Common Errors

**1. Rate Limit Exceeded (429)**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please slow down and try again later.",
  "retry_after": "60 seconds"
}
```

**Frontend Handling**:
```javascript
if (response.status === 429) {
    alert("Too many requests! Please wait 60 seconds and try again.");
    // Disable submit button for 60 seconds
    setTimeout(() => enableSubmit(), 60000);
}
```

**2. Timeout Error (504)**
```json
{
  "detail": "Speech-to-text timed out. Please try a shorter recording."
}
```

**Frontend Handling**:
```javascript
if (error.message.includes("timed out")) {
    alert("Your recording is too long. Please keep it under 2 minutes.");
}
```

**3. No Credentials (404)**
```json
{
  "detail": "No credentials found for user user123."
}
```

**Frontend Handling**:
```javascript
if (error.message.includes("credentials")) {
    // Redirect to OAuth connection page
    window.location.href = `${API_BASE_URL}/auth/twitter/login`;
}
```

**4. Invalid Audio Format (500)**
```json
{
  "detail": "Error transcribing audio: Unsupported format"
}
```

**Frontend Handling**:
```javascript
// Only allow specific formats
<input 
    type="file" 
    accept="audio/wav,audio/mp3,audio/webm,audio/ogg"
    onChange={handleAudioUpload}
/>
```

---

## 📱 Complete React Component Example

```jsx
import React, { useState } from 'react';

const VoiceToPostApp = () => {
    const [audioFile, setAudioFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [posts, setPosts] = useState([]);
    const [images, setImages] = useState([]);
    const [selectedPost, setSelectedPost] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    
    const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:7860";
    
    const handleAudioChange = (e) => {
        const file = e.target.files[0];
        if (file && file.size > 10 * 1024 * 1024) {
            alert("File too large! Maximum 10MB allowed.");
            return;
        }
        setAudioFile(file);
    };
    
    const generatePost = async () => {
        if (!audioFile) {
            alert("Please select an audio file first!");
            return;
        }
        
        setLoading(true);
        
        try {
            const formData = new FormData();
            formData.append('audio_file', audioFile);
            formData.append('tone', 'professional');
            formData.append('platform', 'twitter');
            formData.append('user_id', 'demo_user');
            formData.append('image_method', 'stock');
            formData.append('num_image_options', 3);
            
            const response = await fetch(`${API_BASE_URL}/generate-post-with-image`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Generation failed");
            }
            
            const data = await response.json();
            
            setPosts(data.variations);
            setImages(data.images || []);
            setSelectedPost(data.variations[0]);
            setSelectedImage(data.images?.[0]);
            
            alert("✅ Post generated successfully!");
            
        } catch (error) {
            console.error("Error:", error);
            
            if (error.message.includes("timeout")) {
                alert("⏱️ Request timed out. Please try a shorter recording.");
            } else if (error.message.includes("rate_limit")) {
                alert("🚫 Too many requests. Please wait a minute.");
            } else {
                alert("❌ Error: " + error.message);
            }
        } finally {
            setLoading(false);
        }
    };
    
    const publishPost = async () => {
        if (!selectedPost) {
            alert("Please generate a post first!");
            return;
        }
        
        setLoading(true);
        
        try {
            const formData = new FormData();
            formData.append('platform', 'twitter');
            formData.append('post_text', selectedPost.text);
            formData.append('user_id', 'demo_user');
            
            const response = await fetch(`${API_BASE_URL}/publish-post`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === "success") {
                alert(`🎉 Published to ${data.platform}!\nURL: ${data.url}`);
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            if (error.message.includes("credentials")) {
                alert("⚠️ Please connect your Twitter account first!");
                window.open(`${API_BASE_URL}/auth/twitter/login`, '_blank');
            } else {
                alert("❌ Publishing failed: " + error.message);
            }
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="app">
            <h1>Voice-To-Post Generator</h1>
            
            {/* Audio Upload */}
            <div className="upload-section">
                <input 
                    type="file" 
                    accept="audio/*"
                    onChange={handleAudioChange}
                    disabled={loading}
                />
                <button onClick={generatePost} disabled={loading || !audioFile}>
                    {loading ? "Generating..." : "Generate Post"}
                </button>
            </div>
            
            {/* Posts */}
            {posts.length > 0 && (
                <div className="posts-section">
                    <h2>Generated Posts (Select One)</h2>
                    {posts.map((post, index) => (
                        <div 
                            key={index}
                            className={`post ${selectedPost === post ? 'selected' : ''}`}
                            onClick={() => setSelectedPost(post)}
                        >
                            <p>{post.text}</p>
                            <span className="score">Score: {post.score.toFixed(3)}</span>
                        </div>
                    ))}
                </div>
            )}
            
            {/* Images */}
            {images.length > 0 && (
                <div className="images-section">
                    <h2>Select an Image</h2>
                    <div className="image-grid">
                        {images.map((img, index) => (
                            <img
                                key={index}
                                src={`data:image/jpeg;base64,${img.image_base64}`}
                                alt={`Option ${index + 1}`}
                                className={selectedImage === img ? 'selected' : ''}
                                onClick={() => setSelectedImage(img)}
                            />
                        ))}
                    </div>
                </div>
            )}
            
            {/* Publish */}
            {selectedPost && (
                <div className="publish-section">
                    <button onClick={publishPost} disabled={loading}>
                        {loading ? "Publishing..." : "Publish to Twitter"}
                    </button>
                </div>
            )}
        </div>
    );
};

export default VoiceToPostApp;
```

---

## 🎯 Testing Checklist for Frontend Team

### Basic Tests
- [ ] `/` endpoint returns status
- [ ] Audio upload works (< 10MB)
- [ ] Post generation returns 5 variations
- [ ] Posts are sorted by score
- [ ] Refinement works
- [ ] Hashtag suggestions work
- [ ] Thread generation works

### Error Handling
- [ ] Large file rejected (>10MB)
- [ ] Timeout handled gracefully
- [ ] Rate limit shows proper message
- [ ] Network errors caught
- [ ] Missing credentials handled

### OAuth Flow
- [ ] Twitter login redirect works
- [ ] Callback URL handled
- [ ] User ID captured
- [ ] Publishing works after OAuth

---

## 📞 Backend Contact Points

**Health Check**: `GET /`  
**System Info**: `GET /system/info`  
**Vector Stats**: `GET /vector-store/stats`

**Support**: Check `BACKEND_ISSUES.md` for known issues

---

## ✅ FINAL ANSWER: YES, IT'S READY!

### What Works Perfectly
✅ All 20+ endpoints functional  
✅ Error responses standardized  
✅ Rate limiting active  
✅ CORS enabled  
✅ Authentication optional  
✅ Comprehensive examples provided  

### Minor Recommendations (Not Blocking)
⚠️ Add file size validation (10MB max)  
⚠️ Consider image URLs instead of base64 for mobile  
⚠️ Configure CORS for production domains  

**Your frontend team can start integration TODAY!** 🚀

---

**Last Updated**: 2026-08-23  
**API Version**: 2.0.0  
**Status**: Production Ready
