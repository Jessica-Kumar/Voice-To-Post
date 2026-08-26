import re
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLM instance
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    top_p=0.2
)


async def generate_thread(
    transcript: str,
    platform: str,
    tone: str,
    max_posts: int = 5,
    context: str = ""
) -> List[Dict[str, str]]:
    """
    Generate a multi-post thread from a longer transcript.

    Args:
        transcript: The voice transcript (can be long)
        platform: Target platform (twitter, linkedin)
        tone: Desired tone
        max_posts: Maximum number of posts in thread
        context: Optional user context

    Returns:
        List of posts with thread numbering
    """

    # Character limits per platform
    char_limits = {
        "twitter": 280,
        "linkedin": 1300,  # LinkedIn allows longer posts
        "x": 280
    }

    char_limit = char_limits.get(platform.lower(), 280)

    # Thread generation prompt
    THREAD_PROMPT = PromptTemplate.from_template(
        """You are an expert at creating engaging social media threads.

Your task: Break down the following content into a cohesive {platform} thread.

INPUT CONTENT:
{transcript}

USER CONTEXT (if available):
{context}

REQUIREMENTS:
1. Create {max_posts} connected posts that form a complete narrative
2. Each post MUST be under {char_limit} characters (strictly enforce this)
3. Maintain {tone} tone throughout
4. Start post 1 with a HOOK that grabs attention
5. Each post should stand alone but connect to the next
6. End the thread with a strong conclusion or call-to-action
7. Use 1-2 relevant emojis per post
8. Add 2-3 hashtags ONLY in the final post

OUTPUT FORMAT (CRITICAL):
Return ONLY a JSON array. Each object must have "post_number" and "text" keys.
Do NOT use markdown code blocks. Return raw JSON array directly.

[
  {{"post_number": 1, "text": "Post 1 content here..."}},
  {{"post_number": 2, "text": "Post 2 content here..."}},
  {{"post_number": 3, "text": "Post 3 content here..."}}
]

Generate exactly {max_posts} posts now:"""
    )

    chain = THREAD_PROMPT | llm | StrOutputParser()

    try:
        raw_result = await chain.ainvoke({
            "transcript": transcript,
            "platform": platform,
            "tone": tone,
            "max_posts": max_posts,
            "char_limit": char_limit,
            "context": context or "None"
        })

        # Extract JSON array
        import json
        match = re.search(r'\[.*\]', raw_result, re.DOTALL)

        if not match:
            raise ValueError("No JSON array found in response")

        clean_json = match.group(0)
        thread_posts = json.loads(clean_json)

        # Add thread numbering (1/5, 2/5, etc.)
        total = len(thread_posts)
        for i, post in enumerate(thread_posts, 1):
            if "text" in post:
                # Replace literal \n with actual newlines
                post["text"] = post["text"].replace("\\n", "\n")

                # Add thread number at the end
                post["text"] = f"{post['text']}\n\n({i}/{total})"
                post["post_number"] = i

        return thread_posts[:max_posts]

    except Exception as e:
        print(f"Thread generation error: {e}")
        # Fallback: simple splitting
        return _fallback_thread_split(transcript, platform, max_posts)


def _fallback_thread_split(transcript: str, platform: str, max_posts: int) -> List[Dict[str, str]]:
    """
    Fallback: Split transcript into chunks if AI generation fails.
    """
    char_limit = 250 if platform.lower() in ["twitter", "x"] else 1200

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', transcript)

    threads = []
    current_post = ""
    post_num = 1

    for sentence in sentences:
        if len(current_post) + len(sentence) + 1 < char_limit:
            current_post += sentence + " "
        else:
            if current_post:
                threads.append({
                    "post_number": post_num,
                    "text": current_post.strip()
                })
                post_num += 1
            current_post = sentence + " "

        if post_num > max_posts:
            break

    # Add last post
    if current_post and post_num <= max_posts:
        threads.append({
            "post_number": post_num,
            "text": current_post.strip()
        })

    # Add thread numbering
    total = len(threads)
    for post in threads:
        post["text"] += f"\n\n({post['post_number']}/{total})"

    return threads


async def generate_multi_platform_posts(
    transcript: str,
    platforms: List[str],
    tone: str,
    context: str = ""
) -> Dict[str, Dict[str, str]]:
    """
    Generate optimized posts for multiple platforms simultaneously.

    Args:
        transcript: Voice transcript
        platforms: List of platforms ["twitter", "linkedin", "discord"]
        tone: Desired tone
        context: User context

    Returns:
        Dictionary with platform as key and post data as value
    """
    from generation_service import generate_post_rag

    results = {}

    for platform in platforms:
        try:
            # Generate 3 variations per platform
            variations = await generate_post_rag(
                transcript=transcript,
                retrieved_context=[{"text": context, "distance": 0.0}] if context else [],
                tone=tone,
                platform=platform,
                num_variations=3
            )

            # Return the best one (first one is already sorted by score in main.py)
            if variations:
                results[platform] = {
                    "text": variations[0].get("text", ""),
                    "platform": platform
                }
        except Exception as e:
            print(f"Error generating for {platform}: {e}")
            results[platform] = {
                "text": f"Error generating post for {platform}",
                "platform": platform,
                "error": str(e)
            }

    return results


async def smart_hashtag_suggestions(post_text: str, platform: str, num_hashtags: int = 5) -> List[str]:
    """
    Generate relevant hashtag suggestions for a post.

    Args:
        post_text: The post content
        platform: Target platform
        num_hashtags: Number of hashtags to suggest

    Returns:
        List of hashtag suggestions (without # symbol)
    """
    HASHTAG_PROMPT = PromptTemplate.from_template(
        """You are a social media expert specializing in hashtag strategy.

POST CONTENT:
{post_text}

PLATFORM: {platform}

Generate {num_hashtags} highly relevant, trending-style hashtags for this post.

RULES:
1. Hashtags should be specific to the post content
2. Mix of popular and niche hashtags
3. {platform}-appropriate (Twitter: shorter, LinkedIn: professional)
4. No spaces in hashtags
5. Title case (e.g., #SocialMedia not #socialmedia)

OUTPUT: Return ONLY a comma-separated list of hashtags WITHOUT the # symbol.
Example: SocialMedia, ContentCreation, DigitalMarketing

Generate hashtags now:"""
    )

    chain = HASHTAG_PROMPT | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "post_text": post_text,
            "platform": platform,
            "num_hashtags": num_hashtags
        })

        # Parse comma-separated hashtags
        hashtags = [tag.strip().replace('#', '') for tag in result.split(',')]
        hashtags = [tag for tag in hashtags if tag]  # Remove empty

        return hashtags[:num_hashtags]

    except Exception as e:
        print(f"Hashtag generation error: {e}")
        # Fallback: extract keywords from post
        return _extract_hashtags_from_text(post_text, num_hashtags)


def _extract_hashtags_from_text(text: str, num: int = 5) -> List[str]:
    """Extract potential hashtags from text as fallback."""
    # Remove special characters and split
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)

    # Title case
    hashtags = [word.capitalize() for word in words[:num]]

    return hashtags
