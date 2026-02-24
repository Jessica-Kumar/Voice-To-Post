import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

# Retrieve API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.8,
    convert_system_message_to_human=True
)

# 🔥 CLEANED & STRUCTURED PROMPT
POST_GENERATION_PROMPT = PromptTemplate.from_template(
"""
You are a professional Social Media Content Strategist.

TASK:
Generate EXACTLY 5 distinct social media post variations.

STRICT REQUIREMENTS:
- Each post MUST contain at least 2 relevant hashtags.
- Each post MUST contain at least 2 relevant emojis.
- Do NOT use forbidden words: spam, hate, violence, scam.
- Keep content deeply relevant to transcript and context.
- Vary structure between posts (question, insight, CTA, bold statement, etc.)
- No repeated sentences between variations.

Target Tone: {tone}

Context:
{context}

Transcript:
{transcript}

IMPORTANT:
Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.

FORMAT:
[
  {"text": "Post 1 here"},
  {"text": "Post 2 here"},
  {"text": "Post 3 here"},
  {"text": "Post 4 here"},
  {"text": "Post 5 here"}
]
"""
)


async def generate_post_rag(transcript: str, retrieved_context: list, tone: str) -> list:
    """
    Generates 5 structured post variations using RAG + optional live news.
    """

    formatted_context = format_context(retrieved_context)

    # 🔹 NewsAPI Enhancement
    news_context = ""
    if NEWS_API_KEY:
        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            query = transcript[:50]
            headlines = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=3
            )

            if headlines['status'] == 'ok' and headlines['totalResults'] > 0:
                news_context = "\n\nRelevant Live News:\n"
                news_context += "\n".join(
                    [f"- {a['title']}" for a in headlines['articles']]
                )
        except Exception as e:
            print(f"NewsAPI Error: {e}")

    final_context = formatted_context + news_context

    rag_chain = POST_GENERATION_PROMPT | llm | StrOutputParser()

    try:
        raw_result = await rag_chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone
        })

        # Clean markdown fences if model accidentally adds them
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(clean_json)

        # Validate structure
        if isinstance(parsed, list) and len(parsed) == 5:
            return parsed
        else:
            raise ValueError("Invalid JSON structure")

    except Exception as e:
        print(f"RAG Parsing Error: {e}")

        # 🔥 Safe fallback
        fallback = [
            {"text": f"AI generation fallback. Please try again. 🚀 #VoiceToPost #AI"}
            for _ in range(5)
        ]

        return fallback


def format_context(vector_results: list) -> str:
    if not vector_results:
        return "No specific past context found."

    context_lines = [f"- {res['text']}" for res in vector_results]
    return "\n".join(context_lines)