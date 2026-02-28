import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ✅ Stable LLM with low temperature and top_p – prevents hallucinations and ensures factual output
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    top_p=0.1
)

# 🔥 Single, highly‑instructive prompt (no separate system message)
STRICT_PROMPT = PromptTemplate.from_template(
    """You are a professional Social Media Strategist for a B.Tech CSE student at KIET Group of Institutions.
Your task is to generate EXACTLY 5 distinct, high‑quality social media posts based on the following inputs.

Target Platform: {platform}
Target Tone: {tone}
Context from user's profile and past content: {context}
User's voice transcript (topic/idea): {transcript}

CRITICAL GROUNDING RULES:
- Every post MUST be derived **exclusively** from the provided Context or Transcript. Do not invent facts.
- Use specific terminology from the user's background: 'KIET Group of Institutions', 'B.Tech CSE', etc.
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
    # Format the vector store results into a readable context block
    formatted_context = _format_context(retrieved_context)

    # Optional live news enrichment (if available)
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

    # Build the chain with the single strict prompt
    chain = STRICT_PROMPT | llm | StrOutputParser()

    try:
        raw_result = await chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone,
            "platform": platform
        })

        # Clean potential markdown code fences
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)

        if isinstance(parsed, list) and len(parsed) == 5:
            return parsed
        else:
            raise ValueError("Invalid JSON structure")

    except Exception as e:
        print(f"RAG Parsing Error: {e}")
        # Safe fallback – 5 simple posts
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