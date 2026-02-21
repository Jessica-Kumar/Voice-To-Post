import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from newsapi import NewsApiClient

# Retrieve the API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the environment.")
if not NEWS_API_KEY:
    print("WARNING: NEWS_API_KEY not found. News context will be disabled.")

# Initialize the NewsAPI Client
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None

# Initialize the Gemini model via Langchain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Note: use appropriate gemini model, flash is faster, pro is smarter
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# Create the prompt template for the RAG generation
POST_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are an expert social media manager and content creator.

Given the following raw thoughts (transcribed from audio) and context retrieved from the user's past posts or knowledge base, 
generate an engaging, native-feeling social media post. 
Make sure the tone is consistent with the context provided, if any. 
Ensure it's highly readable, uses appropriate emojis, and has relevant hashtags.

Context from past posts:
{context}

Raw Thoughts (Audio Transcript):
{transcript}

Generated Social Media Post:"""
)

def format_context(vector_results: list) -> str:
    """Helper formatting function for the search results"""
    if not vector_results:
        return "No specific past context found."
        
    context_lines = [f"- {res['text']}" for res in vector_results]
    return "\n".join(context_lines)

async def generate_post_rag(transcript: str, retrieved_context: list) -> str:
    """
    Takes transcribed text and FAISS search results to generate a post via Gemini.
    
    Args:
        transcript (str): The transcribed thoughts from the user.
        retrieved_context (list): The list of dictionaries returned from vector_store.search_index
        
    Returns:
        str: The generated social media post.
    """
    formatted_context = format_context(retrieved_context)
    
    # --- NewsAPI RAG Enhancement ---
    # We will search NewsAPI for top headlines based on the transcript's first few words or entirely
    # and append the top 3 items to the contextual prompt.
    news_context = ""
    if newsapi:
        try:
            # Query NewsAPI (in a real app, you might extract keywords first)
            query = transcript[:50] # using first 50 chars as a rudimentary query
            top_headlines = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=3)
            
            if top_headlines['status'] == 'ok' and top_headlines['totalResults'] > 0:
                articles = top_headlines['articles']
                news_context = "\n\nRelevant Live News context:\n"
                for article in articles:
                    news_context += f"- {article['title']}: {article['description']}\n"
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            
    # Combine FAISS database context with Live NewsAPI context
    final_context = formatted_context + news_context
    
    # LangChain Pipeline (LCEL - LangChain Expression Language)
    rag_chain = (
        {"context": lambda x: x["context"], "transcript": lambda x: x["transcript"]}
        | POST_GENERATION_PROMPT
        | llm
        | StrOutputParser()
    )
    
    try:
        # Invoke the chain
        result = await rag_chain.ainvoke({
            "context": final_context,
            "transcript": transcript
        })
        return result
        
    except Exception as e:
        print(f"Error in RAG generation: {e}")
        return f"Error generating post: {str(e)}"
