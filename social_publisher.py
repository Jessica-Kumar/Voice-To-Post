import httpx
from typing import Optional
from database import SessionLocal, SocialCreds, decrypt_secret

async def publish_to_platform(platform: str, post_content: str) -> dict:
    """
    Retrieves stored credentials and attempts to publish the generated post 
    to the requested social media platform.
    
    Args:
        platform (str): The target platform ("twitter" or "linkedin").
        post_content (str): The text content of the post.
        
    Returns:
        dict: A status dictionary containing success/failure and relevant messages.
    """
    platform = platform.lower()
    
    if platform not in ["twitter", "linkedin"]:
        return {"status": "error", "message": f"Unsupported platform: {platform}"}

    # Retrieve credentials from the SQLite database
    db = SessionLocal()
    creds = db.query(SocialCreds).filter(SocialCreds.platform == platform).first()
    db.close()
    
    if not creds:
        return {
            "status": "error", 
            "message": f"No credentials found for {platform}. Please save your API keys first."
        }
        
    client_id = creds.client_id
    client_secret = decrypt_secret(creds.encrypted_secret)
    
    # ---------------------------------------------------------
    # Mocking the actual publishing logic for demonstration.
    # In a real app, this would involve OAuth2 token exchange
    # (sometimes requiring an explicit user callback flow) 
    # followed by a POST request to the platform's API.
    # ---------------------------------------------------------
    
    try:
        async with httpx.AsyncClient() as client:
            if platform == "twitter":
                # Simulated connection to Twitter/X v2 API
                # URL: https://api.twitter.com/2/tweets
                # payload = {"text": post_content}
                print(f"[Twitter Publisher] Authenticating with Client ID: {client_id[:5]}...")
                print(f"[Twitter Publisher] Posting: {post_content}")
                
            elif platform == "linkedin":
                # Simulated connection to LinkedIn UGC Post API
                # URL: https://api.linkedin.com/v2/ugcPosts
                print(f"[LinkedIn Publisher] Authenticating with Client ID: {client_id[:5]}...")
                print(f"[LinkedIn Publisher] Posting: {post_content}")
                
        # Simulate a successful API response
        return {
            "status": "success",
            "message": f"Successfully published to {platform.capitalize()}!",
            "platform": platform
        }
        
    except Exception as e:
        print(f"Error publishing to {platform}: {e}")
        return {
            "status": "error",
            "message": f"Failed to publish to {platform}: {str(e)}"
        }
