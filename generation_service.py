import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient
from langchain_core.messages import SystemMessage

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ✅ Lower temperature (0.2) and top_p (0.1) for deterministic, grounded output
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    top_p=0.1,
    convert_system_message_to_human=True
)

# 🔥 Updated System Prompt (the "Anchor" Strategy)
SYSTEM_PROMPT = """You are a specialized Social Media Engineer for a B.Tech CSE student.
Technical Parameters:
- Temperature: 0.2 (keep factual, avoid hallucinations)
- Top_P: 0.1 (use only the most professional vocabulary)
- Presence Penalty: 0.0 (stay on the given context)

Instructions:
1. Groundedness: Every claim must be 100% derived from the provided Vector Context (Exhibit A) or the Voice Transcript.
2. Contextual Mirroring: Use the specific terminology from the user's background (e.g., 'KIET Group of Institutions', 'B.Tech CSE') to minimize semantic distance.
3. Score Optimization: Do not use generic AI greetings. Start directly with the hook to keep the relevance score high."""

# 🔥 User prompt template – now only LinkedIn and Twitter
USER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """TASK: Generate EXACTLY 5 distinct social media post variations.
Target Platform: {platform}
Target Tone: {tone}
Context: {context}
Transcript: {transcript}

PLATFORM RULES:
- Twitter/X: Strictly under 280 characters. Short, punchy.
- LinkedIn: Detailed, professional, line breaks, networking focus.

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

# Combine system and user prompt
def create_full_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE)
    ])

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

    # Build the chat prompt with system and user messages
    prompt = create_full_prompt()
    chain = prompt | llm | StrOutputParser()

    try:
        raw_result = await chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone,
            "platform": platform
        })
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        if isinstance(parsed, list) and len(parsed) == 5:
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