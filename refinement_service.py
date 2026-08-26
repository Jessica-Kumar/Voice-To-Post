from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    top_p=0.2
)


async def refine_post(
    original_post: str,
    refinement_type: str,
    platform: str = "twitter",
    custom_instruction: str = None
) -> str:
    """
    Refine an existing post based on user feedback.

    Args:
        original_post: The original post text
        refinement_type: Type of refinement (see REFINEMENT_TYPES)
        platform: Target platform
        custom_instruction: Optional custom refinement instruction

    Returns:
        Refined post text
    """

    REFINEMENT_TYPES = {
        "shorten": "Make this post shorter and more concise while keeping the core message. Remove unnecessary words.",
        "lengthen": "Expand this post with more details, context, and supporting points. Make it more comprehensive.",
        "more_formal": "Rewrite this post in a more formal, professional tone. Use business language.",
        "more_casual": "Rewrite this post in a more casual, conversational tone. Make it friendly and approachable.",
        "add_humor": "Add humor and wit to this post. Make it entertaining while keeping the message.",
        "add_hooks": "Improve the opening line to be more attention-grabbing. Create a strong hook.",
        "add_cta": "Add a clear call-to-action at the end. Encourage engagement (comment, share, etc.).",
        "remove_jargon": "Simplify this post by removing jargon and technical terms. Make it accessible to everyone.",
        "add_emojis": "Add 2-3 relevant emojis to make the post more engaging and visual.",
        "remove_emojis": "Remove all emojis and make the post text-only.",
        "add_hashtags": "Add 3 relevant hashtags at the end of this post.",
        "more_professional": "Make this post more professional and authoritative. Suitable for LinkedIn.",
        "more_engaging": "Make this post more engaging with questions, emojis, and conversational language."
    }

    # Get refinement instruction
    if custom_instruction:
        instruction = custom_instruction
    elif refinement_type in REFINEMENT_TYPES:
        instruction = REFINEMENT_TYPES[refinement_type]
    else:
        instruction = refinement_type  # Use as custom instruction

    # Platform constraints
    char_limits = {
        "twitter": 280,
        "x": 280,
        "linkedin": 1300,
        "discord": 2000,
        "medium": 5000
    }
    char_limit = char_limits.get(platform.lower(), 280)

    REFINEMENT_PROMPT = PromptTemplate.from_template(
        """You are an expert social media editor.

ORIGINAL POST:
{original_post}

REFINEMENT TASK:
{instruction}

PLATFORM: {platform} (Character limit: {char_limit})

REQUIREMENTS:
1. Apply the refinement while staying under {char_limit} characters
2. Maintain the core message and intent
3. Keep it suitable for {platform}
4. Return ONLY the refined post text, nothing else
5. No explanations, no quotes, just the refined content

Refined post:"""
    )

    chain = REFINEMENT_PROMPT | llm | StrOutputParser()

    try:
        refined = await chain.ainvoke({
            "original_post": original_post,
            "instruction": instruction,
            "platform": platform,
            "char_limit": char_limit
        })

        # Clean up
        refined = refined.strip()
        refined = refined.replace("\\n", "\n")

        # Remove quotes if LLM added them
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1]
        if refined.startswith("'") and refined.endswith("'"):
            refined = refined[1:-1]

        return refined

    except Exception as e:
        print(f"Post refinement error: {e}")
        return original_post  # Return original on error


async def analyze_post_quality(post_text: str, platform: str) -> dict:
    """
    Analyze post quality and provide improvement suggestions.

    Args:
        post_text: The post to analyze
        platform: Target platform

    Returns:
        Dictionary with quality metrics and suggestions
    """
    ANALYSIS_PROMPT = PromptTemplate.from_template(
        """You are a social media content quality analyst.

POST TO ANALYZE:
{post_text}

PLATFORM: {platform}

Provide a detailed quality analysis in the following JSON format:
{{
  "readability_score": <1-10>,
  "engagement_potential": <1-10>,
  "clarity_score": <1-10>,
  "tone_appropriateness": <1-10>,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}

Return ONLY valid JSON, no markdown, no extra text.

Analysis:"""
    )

    chain = ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        import json
        import re

        result = await chain.ainvoke({
            "post_text": post_text,
            "platform": platform
        })

        # Extract JSON
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            analysis = json.loads(match.group(0))
            return analysis
        else:
            raise ValueError("No JSON found in analysis")

    except Exception as e:
        print(f"Post analysis error: {e}")
        # Return default analysis
        return {
            "readability_score": 5,
            "engagement_potential": 5,
            "clarity_score": 5,
            "tone_appropriateness": 5,
            "strengths": ["Content provided"],
            "weaknesses": ["Analysis unavailable"],
            "suggestions": ["Try refining the post for better engagement"]
        }


async def batch_refine_posts(
    posts: list,
    refinement_type: str,
    platform: str = "twitter"
) -> list:
    """
    Refine multiple posts at once.

    Args:
        posts: List of post texts
        refinement_type: Type of refinement to apply
        platform: Target platform

    Returns:
        List of refined posts
    """
    refined_posts = []

    for post in posts:
        try:
            refined = await refine_post(post, refinement_type, platform)
            refined_posts.append(refined)
        except Exception as e:
            print(f"Error refining post: {e}")
            refined_posts.append(post)  # Keep original on error

    return refined_posts


# Available refinement types for API documentation
AVAILABLE_REFINEMENTS = {
    "shorten": "Reduce word count while keeping the message",
    "lengthen": "Add more detail and context",
    "more_formal": "Professional business tone",
    "more_casual": "Friendly conversational tone",
    "add_humor": "Make it funny and entertaining",
    "add_hooks": "Improve opening line",
    "add_cta": "Add call-to-action",
    "remove_jargon": "Simplify technical language",
    "add_emojis": "Add relevant emojis",
    "remove_emojis": "Remove all emojis",
    "add_hashtags": "Add relevant hashtags",
    "more_professional": "LinkedIn-appropriate",
    "more_engaging": "Increase engagement potential"
}
