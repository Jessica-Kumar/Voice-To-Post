import os
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# FIX: Raised temperature 0.2→0.9 and top_p 0.1→0.95
# At 0.2/0.1, all 5 posts were nearly identical in structure → same score.
# Higher temperature produces genuinely different lengths, emoji usage,
# hashtag counts, and hooks — which scoring.py can now differentiate.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.9,
    top_p=0.95
)

STRICT_PROMPT = PromptTemplate.from_template(
    """You are an elite Social Media Ghostwriter and Strategist.

Your objective is to generate EXACTLY 5 DISTINCT social media posts based ONLY on the provided inputs.

INPUT DATA:
- Target Platform: {platform}
- Target Tone: {tone}
- Context (User's profile, bio, and past posts): {context}
- Voice Transcript (The core topic/idea): {transcript}

CRITICAL ANTI-HALLUCINATION INVARIANTS:
1. ZERO FABRICATION: Do not invent numbers, job titles, companies, or names. Use ONLY facts from Context or Transcript.
2. THE GHOSTWRITING RULE: Adopt the user's vocabulary and sentence structure from their Context.
3. THE DISCONNECT FALLBACK: If Transcript is unrelated to Context, write a professional post focused purely on the Transcript topic.
4. NO FLUFF: No generic AI phrases. Start every post with a strong, scroll-stopping hook.

PLATFORM-SPECIFIC GUIDELINES:
- **Twitter/X**: Strictly ≤ 280 characters. Short, punchy, impactful.
- **LinkedIn**: 150–600 words. Use line breaks. Professional and insightful.

DIVERSITY REQUIREMENT — THIS IS CRITICAL:
Each of the 5 posts MUST be meaningfully different from the others:
- Post 1: Hook with a bold statement. 2–3 hashtags. 1–2 emojis.
- Post 2: Opens with a question. 1 hashtag only. No emojis.
- Post 3: Storytelling / narrative style. 2 hashtags. 2–3 emojis.
- Post 4: Data or insight-driven. End with a CTA ("Comment below", "Share if you agree", etc.). 3 hashtags.
- Post 5: Short and punchy (even on LinkedIn — max 3 sentences). 1–2 hashtags. 1 emoji.

STRICT OUTPUT FORMAT:
Return ONLY a valid JSON array of exactly 5 objects, each with a single "text" key.
Do NOT wrap in markdown (no ```json). Return raw JSON only.

[
  {{"text": "<Post 1>"}},
  {{"text": "<Post 2>"}},
  {{"text": "<Post 3>"}},
  {{"text": "<Post 4>"}},
  {{"text": "<Post 5>"}}
]
"""
)


async def generate_post_rag(
    transcript: str,
    retrieved_context: list,
    tone: str,
    platform: str,
    num_variations: int = 5
) -> list:

    formatted_context = _format_context(retrieved_context)

    # Optional live news enrichment
    news_context = ""
    if NEWS_API_KEY:
        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            query = transcript[:50]
            headlines = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=3)
            if headlines['status'] == 'ok' and headlines['totalResults'] > 0:
                news_context = "\n\nRelevant Live News:\n" + "\n".join(
                    [f"- {a['title']}" for a in headlines['articles']]
                )
        except Exception as e:
            print(f"NewsAPI Error: {e}")

    final_context = formatted_context + news_context

    chain = STRICT_PROMPT | llm | StrOutputParser()

    try:
        raw_result = await chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone,
            "platform": platform
        })

        print(f"Raw LLM output: {raw_result[:500]}")

        # Regex extraction — find the first JSON array
        match = re.search(r'\[.*\]', raw_result, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response")

        clean_json = match.group(0)
        parsed = json.loads(clean_json)

        if isinstance(parsed, list):
            for post in parsed:
                if "text" in post:
                    post["text"] = post["text"].replace("\\n", "\n")
            return parsed[:5]
        else:
            raise ValueError("Parsed JSON is not a list")

    except Exception as e:
        print(f"RAG Parsing Error: {e}")
        fallback = [
            {"text": f"AI generation fallback. Please try again. 🚀 #VoiceToPost #AI"}
            for _ in range(5)
        ]
        return fallback


def _format_context(vector_results: list) -> str:
    if not vector_results:
        return "No specific past context found."
    context_lines = [f"- {res['text']}" for res in vector_results]
    return "\n".join(context_lines)
