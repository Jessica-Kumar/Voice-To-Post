import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from newsapi import NewsApiClient

# ... (Keep your existing API Key checks and NewsAPI initialization) ...

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Updated to current stable flash model
    google_api_key=GEMINI_API_KEY,
    temperature=0.8 # Slightly higher for better variation variety
)

# UPGRADED PROMPT: Now requests 5 variations in a specific JSON format
POST_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are an expert social media manager. 
Based on the following context, news, and raw thoughts, generate exactly 5 distinct social media post variations.

Target Tone: {tone}

Context:
{context}

Raw Thoughts:
{transcript}

Return the response as a valid JSON array of objects with the following structure:
[
  {{"text": "post content 1"}},
  {{"text": "post content 2"}},
  ...
]
Ensure the tone is strictly {tone}. Include relevant emojis and hashtags."""
)

async def generate_post_rag(transcript: str, retrieved_context: list, tone: str) -> list:
    """
    Generates 5 post variations based on transcript, RAG context, and a specific tone.
    """
    formatted_context = format_context(retrieved_context)
    
    # --- NewsAPI RAG Enhancement ---
    news_context = ""
    if newsapi:
        try:
            query = transcript[:50] 
            top_headlines = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=3)
            if top_headlines['status'] == 'ok' and top_headlines['totalResults'] > 0:
                articles = top_headlines['articles']
                news_context = "\n\nRelevant Live News context:\n"
                for article in articles:
                    news_context += f"- {article['title']}: {article['description']}\n"
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            
    final_context = formatted_context + news_context
    
    # Updated Chain to use the new Prompt with Tone
    rag_chain = POST_GENERATION_PROMPT | llm | StrOutputParser()
    
    try:
        # Invoke the chain
        raw_result = await rag_chain.ainvoke({
            "context": final_context,
            "transcript": transcript,
            "tone": tone
        })
        
        # Clean the string to ensure it's valid JSON (removing markdown blocks if LLM adds them)
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        variations = json.loads(clean_json)
        
        return variations # Returns the list of 5 variations for the Android carousel
        
    except Exception as e:
        print(f"Error in RAG generation: {e}")
        return [{"text": f"Error generating posts: {str(e)}"}]