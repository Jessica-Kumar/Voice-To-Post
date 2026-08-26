import os
import asyncio
import httpx
import base64
import io
from typing import Dict, Any, List, Optional
from PIL import Image

# FREE API Keys (to be added to .env)
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Already have this!

# Platform-specific image specifications
IMAGE_SPECS = {
    "twitter": {"width": 1200, "height": 675, "aspect_ratio": "16:9"},
    "linkedin": {"width": 1200, "height": 627, "aspect_ratio": "1.91:1"},
    "instagram": {"width": 1080, "height": 1080, "aspect_ratio": "1:1"},
    "discord": {"width": 1920, "height": 1080, "aspect_ratio": "16:9"},
    "medium": {"width": 1400, "height": 788, "aspect_ratio": "16:9"}
}


async def generate_image_from_post(
    post_text: str,
    platform: str = "twitter",
    method: str = "stock"  # "stock" or "ai"
) -> Dict[str, Any]:
    """
    Generate or fetch an image for a social media post.

    Args:
        post_text: The post content
        platform: Target platform for sizing
        method: "stock" (free stock photos) or "ai" (Hugging Face generation)

    Returns:
        {
            "image_url": str,
            "image_bytes": bytes,
            "thumbnail_url": str,
            "source": str,
            "keywords": List[str]
        }
    """
    # Extract keywords from post
    keywords = await _extract_keywords(post_text)

    if method == "stock":
        # Try Pexels first (higher rate limit), then Unsplash
        try:
            return await _fetch_pexels_image(keywords, platform)
        except Exception as e:
            print(f"Pexels failed: {e}, trying Unsplash...")
            return await _fetch_unsplash_image(keywords, platform)
    elif method == "ai":
        return await _generate_hf_image(post_text, keywords, platform)
    else:
        raise ValueError(f"Unknown method: {method}")


async def _extract_keywords(post_text: str) -> List[str]:
    """Extract 3-5 keywords from post text for image search."""
    # Simple keyword extraction (can be enhanced with NLP)
    import re
    from collections import Counter

    # Remove hashtags, mentions, URLs
    clean_text = re.sub(r'#\w+|@\w+|http\S+|[^\w\s]', '', post_text.lower())

    # Common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

    # Extract words
    words = [w for w in clean_text.split() if len(w) > 3 and w not in stop_words]

    # Get top 5 most common
    word_counts = Counter(words)
    keywords = [word for word, count in word_counts.most_common(5)]

    # Fallback to generic business terms if no keywords
    if not keywords:
        keywords = ["business", "professional", "office", "team", "technology"]

    return keywords[:3]  # Return top 3


async def _fetch_pexels_image(keywords: List[str], platform: str) -> Dict[str, Any]:
    """
    Fetch image from Pexels API (FREE - 200 requests/hour).
    """
    if not PEXELS_API_KEY:
        raise ValueError("PEXELS_API_KEY not set in environment")

    query = " ".join(keywords)
    spec = IMAGE_SPECS[platform]

    # Pexels API endpoint
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": 5,
        "orientation": "landscape" if spec["width"] > spec["height"] else "portrait"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Pexels API error: {response.status_code} - {response.text}")

        data = response.json()

        if not data.get("photos"):
            raise Exception(f"No images found for keywords: {keywords}")

        # Get first photo
        photo = data["photos"][0]

        # Download image
        image_url = photo["src"]["large2x"]  # High quality
        img_response = await client.get(image_url)
        image_bytes = img_response.content

        # Resize to platform specs
        resized_bytes = await _resize_image(image_bytes, spec["width"], spec["height"])

        return {
            "image_url": image_url,
            "image_bytes": resized_bytes,
            "thumbnail_url": photo["src"]["medium"],
            "source": "pexels",
            "photographer": photo["photographer"],
            "photographer_url": photo["photographer_url"],
            "keywords": keywords
        }


async def _fetch_unsplash_image(keywords: List[str], platform: str) -> Dict[str, Any]:
    """
    Fetch image from Unsplash API (FREE - 50 requests/hour).
    """
    if not UNSPLASH_ACCESS_KEY:
        raise ValueError("UNSPLASH_ACCESS_KEY not set in environment")

    query = " ".join(keywords)
    spec = IMAGE_SPECS[platform]

    # Unsplash API endpoint
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "per_page": 5,
        "orientation": "landscape" if spec["width"] > spec["height"] else "portrait"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Unsplash API error: {response.status_code} - {response.text}")

        data = response.json()

        if not data.get("results"):
            raise Exception(f"No images found for keywords: {keywords}")

        # Get first photo
        photo = data["results"][0]

        # Download image
        image_url = photo["urls"]["regular"]
        img_response = await client.get(image_url)
        image_bytes = img_response.content

        # Resize to platform specs
        resized_bytes = await _resize_image(image_bytes, spec["width"], spec["height"])

        return {
            "image_url": image_url,
            "image_bytes": resized_bytes,
            "thumbnail_url": photo["urls"]["small"],
            "source": "unsplash",
            "photographer": photo["user"]["name"],
            "photographer_url": photo["user"]["links"]["html"],
            "keywords": keywords
        }


async def _generate_hf_image(post_text: str, keywords: List[str], platform: str) -> Dict[str, Any]:
    """
    Generate image using Hugging Face Inference API (FREE with rate limits).
    Uses Stable Diffusion models.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set in environment")

    # Create optimized prompt
    prompt = f"professional high-quality photograph of {', '.join(keywords)}, business style, clean composition, natural lighting, 4k resolution"

    # Hugging Face Inference API
    # Free models: stable-diffusion-v1-5, stable-diffusion-2-1
    model_id = "stabilityai/stable-diffusion-2-1"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "text, watermark, low quality, blurry, distorted",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:  # AI generation takes longer
        response = await client.post(api_url, headers=headers, json=payload)

        if response.status_code != 200:
            # Model might be loading, return error with fallback suggestion
            error_data = response.json()
            if "estimated_time" in error_data:
                raise Exception(f"Model is loading. Estimated time: {error_data['estimated_time']}s. Please try stock images instead.")
            raise Exception(f"HF Inference API error: {response.status_code} - {response.text}")

        # Response is raw image bytes
        image_bytes = response.content

        # Resize to platform specs
        spec = IMAGE_SPECS[platform]
        resized_bytes = await _resize_image(image_bytes, spec["width"], spec["height"])

        return {
            "image_url": None,  # Generated, not from URL
            "image_bytes": resized_bytes,
            "thumbnail_url": None,
            "source": "huggingface_ai",
            "model": model_id,
            "prompt": prompt,
            "keywords": keywords
        }


async def _resize_image(image_bytes: bytes, target_width: int, target_height: int) -> bytes:
    """Resize off the event loop (PIL is CPU-bound)."""
    return await asyncio.to_thread(_resize_image_sync, image_bytes, target_width, target_height)


def _resize_image_sync(image_bytes: bytes, target_width: int, target_height: int) -> bytes:
    """
    Resize image to target dimensions while maintaining aspect ratio.
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (removes alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background

        # Resize with high-quality resampling
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=90, optimize=True)
        return output.getvalue()

    except Exception as e:
        print(f"Image resize error: {e}")
        # Return original if resize fails
        return image_bytes


async def get_multiple_image_options(
    post_text: str,
    platform: str = "twitter",
    num_options: int = 3
) -> List[Dict[str, Any]]:
    """
    Get multiple image options for user to choose from.
    Combines stock photos and AI generation.
    """
    keywords = await _extract_keywords(post_text)
    options = []

    # Try to get mix of sources
    try:
        # Option 1: Pexels
        pexels_img = await _fetch_pexels_image(keywords, platform)
        options.append(pexels_img)
    except Exception as e:
        print(f"Pexels option failed: {e}")

    try:
        # Option 2: Unsplash
        unsplash_img = await _fetch_unsplash_image(keywords, platform)
        options.append(unsplash_img)
    except Exception as e:
        print(f"Unsplash option failed: {e}")

    if HF_TOKEN and len(options) < num_options:
        try:
            # Option 3: AI Generated
            ai_img = await _generate_hf_image(post_text, keywords, platform)
            options.append(ai_img)
        except Exception as e:
            print(f"AI generation option failed: {e}")

    return options[:num_options]


# Helper function to encode image as base64 for API responses
def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes as base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')
