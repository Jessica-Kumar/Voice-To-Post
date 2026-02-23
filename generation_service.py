import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

# Retrieve API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize Gemini with a higher temperature for variety in the 5 variations
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.8,
    convert_system_message_to_human=True # Helps with older API compatibility
)

# UPGRADED PROMPT: Now requests 5 variations in a specific JSON format
POST_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are an expert social media manager and a Strict Security Auditor.

Goal: Generate exactly 5 distinct social media post variations based on the transcript.

Audit Instructions:

Tone Validation: Every post must match the '{tone}' tone perfectly.

Security Check: Do not include any text that is offensive, harmful, or unprofessional.

Self-Correction: Review each variation. If it is low quality, rewrite it immediately to ensure it passes a 0.75 safety score.

Format: Return ONLY a valid JSON array of objects: [{"text": "..."}].

Context: {context}
Transcript: {transcript}"""
)

async def generate_post_rag(transcript: str, retrieved_context: list, tone: str) -> list:
    """
    Upgraded function to support specific Tones and 5-Post variations for the UI carousel.
    """
    formatted_context = format_context(retrieved_context)
    
    # NewsAPI RAG Enhancement
    news_context = ""
    if NEWS_API_KEY:
        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            query = transcript[:50] 
            top_headlines = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=3)
            if top_headlines['status'] == 'ok' and top_headlines['totalResults'] > 0:
                news_context = "\n\nRelevant Live News:\n" + "\n".join([f"- {a['title']}" for a in top_headlines['articles']])
        except Exception as e:
            print(f"NewsAPI Error: {e}")

    final_context = formatted_context + news_context
    
    # Updated Chain to include the Tone parameter
    rag_chain = POST_GENERATION_PROMPT | llm | StrOutputParser()
    
    try:
        # Invoke the chain
        raw_result = await rag_chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone
        })
        
        # Parse the string into a list for the Android carousel
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"RAG Error: {e}")
        return [{"text": f"Error generating posts: {str(e)}"}]


def format_context(vector_results: list) -> str:
    """Helper formatting function for the search results"""
    if not vector_results:
        return "No specific past context found."
        
    context_lines = [f"- {res['text']}" for res in vector_results]
    return "\n".join(context_lines)