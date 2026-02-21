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
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.8 
)

# UPGRADED PROMPT: Now requests 5 variations in a specific JSON format
POST_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are an expert social media manager. 
Based on the following context, news, and raw thoughts, generate exactly 5 distinct social media post variations.

Target Tone: {tone}

Context:
{context}

Raw Thoughts (Audio Transcript):
{transcript}

Return the response ONLY as a valid JSON array of objects with the following structure:
[
  {{"text": "post variation 1 content"}},
  {{"text": "post variation 2 content"}},
  ...
]
Ensure the tone matches '{tone}' exactly. Include relevant emojis and hashtags."""
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
