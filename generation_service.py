import os
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Stable LLM with low temperature and top_p – remains unchanged
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    top_p=0.1
)

# Strict prompt (unchanged)
STRICT_PROMPT = PromptTemplate.from_template(
    """You are an elite, highly logical Social Media Ghostwriter and Strategist.
Your objective is to generate EXACTLY 5 distinct, high-quality social media posts based ONLY on the provided inputs.

INPUT DATA:
- Target Platform: {platform}
- Target Tone: {tone}
- Context (User's profile, bio, and past posts): {context}
- Voice Transcript (The core topic/idea): {transcript}

CRITICAL ANTI-HALLUCINATION INVARIANTS:
1. ZERO FABRICATION: You are strictly forbidden from inventing numbers, job titles, companies, names, or personal anecdotes. Extract facts EXCLUSIVELY from the Context or Transcript.
2. THE GHOSTWRITING RULE: Analyze the 'Context' to identify the user's profession and natural writing style. Adopt their vocabulary and sentence structure perfectly.
3. THE DISCONNECT FALLBACK: If the 'Transcript' topic is completely unrelated to the user's 'Context', do NOT force a bizarre connection. Instead, write a highly professional, objective post focused solely on the 'Transcript' topic.
4. NO FLUFF: Avoid generic AI buzzwords (e.g., "In today's fast-paced digital world"). Start every post immediately with a strong, scroll-stopping hook.

PLATFORM-SPECIFIC GUIDELINES:
- **Twitter/X**: Strictly ≤ 280 characters. Short, punchy, impactful.
- **LinkedIn**: Detailed and professional. Use line breaks for readability. Focused on networking and industry value.

FORMATTING REQUIREMENTS:
- Include exactly 2-3 highly relevant hashtags at the end.
- Integrate 1-2 appropriate emojis naturally (!, ?, 🚀, 💡, 🔥, 🌍).
- Do not include any introductory or concluding conversational text.

STRICT OUTPUT FORMAT (API REQUIREMENT):
You must return ONLY a valid JSON array containing exactly 5 objects. Each object must have a single key named "text".
CRITICAL: Do NOT wrap the JSON in markdown blocks (e.g., no ```json). Return the raw, parseable bracket structure directly.
[
  {{"text": "<First engaging post here>"}},
  {{"text": "<Second engaging post here>"}},
  {{"text": "<Third engaging post here>"}},
  {{"text": "<Fourth engaging post here>"}},
  {{"text": "<Fifth engaging post here>"}}
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
    # Format the vector store results
    formatted_context = _format_context(retrieved_context)

    # Optional live news enrichment
    news_context = ""
    if NEWS_API_KEY:
        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            query = transcript[:50]
            headlines = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=3)
            if headlines['status'] == 'ok' and headlines['totalResults'] > 0:
                news_context = "\n\nRelevant Live News:\n" + "\n".join([f"- {a['title']}" for a in headlines['articles']])
        except Exception as e:
            print(f"NewsAPI Error: {e}")

    final_context = formatted_context + news_context

    # Build chain
    chain = STRICT_PROMPT | llm | StrOutputParser()

    try:
        raw_result = await chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone,
            "platform": platform
        })

        print(f"Raw LLM output: {raw_result[:500]}")  # Debug log

        # 🔥 Regex extraction – find the first JSON array
        match = re.search(r'\[.*\]', raw_result, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response")

        clean_json = match.group(0)
        parsed = json.loads(clean_json)

        # Flexible array handling – accept any list, take first 5
        if isinstance(parsed, list):
            return parsed[:5]   # Return up to 5 posts
        else:
            raise ValueError("Parsed JSON is not a list")

    except Exception as e:
        print(f"RAG Parsing Error: {e}")
        # Safe fallback
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