import os
import json
import re
import asyncio
import google.generativeai as genai
from newsapi import NewsApiClient

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Direct SDK — avoids LangChain version mismatches with newer Gemini models
genai.configure(api_key=GEMINI_API_KEY)

GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.9,
    top_p=0.95,
    max_output_tokens=4096,
)

# Try gemini-2.5-flash first; fall back to gemini-1.5-flash if unavailable
PRIMARY_MODEL   = "gemini-2.5-flash-preview-05-20"
FALLBACK_MODEL  = "gemini-1.5-flash"

PROMPT_TEMPLATE = """You are an elite Social Media Ghostwriter and Strategist.

Generate EXACTLY 5 DISTINCT social media posts based ONLY on the provided inputs.

INPUT DATA:
- Target Platform: {platform}
- Target Tone: {tone}
- Context (User profile, bio, past posts): {context}
- Voice Transcript (core idea): {transcript}

ANTI-HALLUCINATION RULES:
1. ZERO FABRICATION: No invented numbers, names, companies. Only facts from Context or Transcript.
2. GHOSTWRITING: Match the user's vocabulary and style from their Context.
3. DISCONNECT FALLBACK: If transcript is unrelated to context, write a professional post on the transcript topic only.
4. NO FLUFF: No generic AI phrases. Start every post with a strong scroll-stopping hook.

PLATFORM GUIDELINES:
- Twitter/X: strictly ≤280 characters. Short and punchy.
- LinkedIn: 150-600 characters. Professional, use line breaks.

MANDATORY DIVERSITY — each post MUST be structurally different:
- Post 1: Bold statement hook. 2-3 hashtags. 1-2 emojis.
- Post 2: Opens with a question. 1 hashtag only. No emojis.
- Post 3: Storytelling / narrative style. 2 hashtags. 2-3 emojis.
- Post 4: Data or insight-driven. Ends with a CTA (e.g. "Comment below" or "Share if you agree"). 3 hashtags.
- Post 5: Very short and punchy (max 3 sentences even on LinkedIn). 1-2 hashtags. 1 emoji.

STRICT OUTPUT FORMAT:
Return ONLY a raw JSON array. No markdown. No ```json. No explanation. Just the array:
[
  {{"text": "<Post 1 here>"}},
  {{"text": "<Post 2 here>"}},
  {{"text": "<Post 3 here>"}},
  {{"text": "<Post 4 here>"}},
  {{"text": "<Post 5 here>"}}
]"""


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
            headlines = newsapi.get_everything(
                q=transcript[:50], language='en', sort_by='relevancy', page_size=3
            )
            if headlines['status'] == 'ok' and headlines['totalResults'] > 0:
                news_context = "\n\nRelevant Live News:\n" + "\n".join(
                    [f"- {a['title']}" for a in headlines['articles']]
                )
        except Exception as e:
            print(f"NewsAPI Error: {e}")

    final_context = formatted_context + news_context
    prompt = PROMPT_TEMPLATE.format(
        platform=platform,
        tone=tone,
        context=final_context,
        transcript=transcript
    )

    # Try primary model, fall back if it errors
    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=GENERATION_CONFIG
            )

            print(f"Calling Gemini model: {model_name}")

            # Run sync SDK call in thread pool so it doesn't block FastAPI's async loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )

            raw_result = response.text
            print(f"Raw LLM output (first 500): {raw_result[:500]}")

            # Strip markdown fences if model added them despite instructions
            raw_result = re.sub(r'```json\s*', '', raw_result)
            raw_result = re.sub(r'```\s*', '', raw_result)
            raw_result = raw_result.strip()

            # Find JSON array
            match = re.search(r'\[.*\]', raw_result, re.DOTALL)
            if not match:
                print(f"No JSON array found in output from {model_name}. Trying fallback.")
                continue  # try next model

            parsed = json.loads(match.group(0))

            if not isinstance(parsed, list) or len(parsed) == 0:
                print(f"Empty/invalid list from {model_name}. Trying fallback.")
                continue

            # Fix escaped newlines
            for post in parsed:
                if "text" in post:
                    post["text"] = post["text"].replace("\\n", "\n")

            print(f"Success: {len(parsed)} posts from {model_name}")
            return parsed[:5]

        except Exception as e:
            # Print the REAL error — visible in HuggingFace Space logs
            print(f"ERROR from {model_name}: {type(e).__name__}: {e}")
            continue  # try next model

    # Both models failed — return diagnostic post (NOT silent fallback)
    # This makes the error visible in the API response for debugging
    print("CRITICAL: Both Gemini models failed.")
    return [
        {"text": f"[GEMINI ERROR] Both models failed. Check HF Space logs for details. Transcript was: {transcript[:100]}"}
    ]


def _format_context(vector_results: list) -> str:
    if not vector_results:
        return "No specific past context found."
    return "\n".join(f"- {res['text']}" for res in vector_results)
