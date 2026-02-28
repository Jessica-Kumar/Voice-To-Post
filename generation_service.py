import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ✅ Using a real model (gemini-1.5-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.8,
    convert_system_message_to_human=True
)

# 🔥 YOUR EXACT PROMPT (with double braces for JSON)
POST_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are a professional Social Media Content Strategist.
TASK: Generate EXACTLY 5 distinct social media post variations.
Target Platform: {platform}
Target Tone: {tone}
Context: {context}
Transcript: {transcript}
PLATFORM RULES:
- Twitter/X: Strictly under 280 characters. Short, punchy.
- LinkedIn: Detailed, professional, line breaks, networking focus.
- Instagram: Visual descriptions, catchy hooks, hashtags at bottom.
STRICT REQUIREMENTS:
- At least 2 hashtags.
- At least 2 emojis (!, ?, 🚀, 💡, 🔥, 🌍).
- No forbidden words: spam, hate, violence, scam.
- Highly relevant to transcript.
IMPORTANT: Return ONLY valid JSON array. No markdown. No extra text.
[
  {{"text": "<Write actual engaging post 1 here>"}},
  {{"text": "<Write actual engaging post 2 here>"}},
  {{"text": "<Write actual engaging post 3 here>"}},
  {{"text": "<Write actual engaging post 4 here>"}},
  {{"text": "<Write actual engaging post 5 here>"}}
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

    rag_chain = POST_GENERATION_PROMPT | llm | StrOutputParser()

    try:
        raw_result = await rag_chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone,
            "platform": platform
        })
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        if isinstance(parsed, list) and len(parsed) == 5:  # prompt always returns 5
            return parsed
        else:
            raise ValueError("Invalid JSON structure")
    except Exception as e:
        print(f"RAG Parsing Error: {e}")
        fallback = [{"text": f"AI generation fallback. Please try again. 🚀 #VoiceToPost #AI"} for _ in range(5)]
        return fallback

def _format_context(vector_results: list) -> str:
    if not vector_results:
        return "No specific past context found."
    context_lines = [f"- {res['text']}" for res in vector_results]
    return "\n".join(context_lines)