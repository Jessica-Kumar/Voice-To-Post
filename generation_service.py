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
    """You are a professional Social Media Strategist.
Your task is to generate EXACTLY 5 distinct, high-quality social media posts based on the following inputs.

Target Platform: {platform}
Target Tone: {tone}
Context (User's profile, bio, and previous posts): {context}
User's voice transcript (topic/idea): {transcript}

CRITICAL GROUNDING RULES:
- Every post MUST be derived **exclusively** from the provided Context or Transcript. Do not invent facts.
- Identify the user's profession, background, or identity from the 'Context' (which includes their profile info and previous posts) and weave that naturally into the post.
- If previous posts are present in the Context, study them to match the user's natural writing style, vocabulary, and formatting preferences.
- Start each post directly with the hook – no generic AI greetings like "Here's a post".
- Stay factual and professional; avoid fluff.

PLATFORM GUIDELINES:
- **Twitter/X**: Strictly ≤ 280 characters, short, punchy.
- **LinkedIn**: Detailed, professional, can include line breaks, networking focused.

FORMATTING REQUIREMENTS (to pass the safety gate):
- Include at least 2 relevant hashtags.
- Include at least 2 emojis (!, ?, 🚀, 💡, 🔥, 🌍 are allowed).
- No forbidden words: spam, hate, violence, scam.

OUTPUT FORMAT:
Return ONLY a valid JSON array with exactly 5 objects, each having a key "text". Do NOT include markdown or any extra text.
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