import asyncio
import tweepy
import httpx
from database import decrypt_secret

async def publish_to_platform(platform: str, post_content: str, creds) -> dict:
    """
    Publish to the specified platform using the stored OAuth tokens.
    creds is an ORM object from SocialCreds (must contain decrypted token fields).
    """
    platform = platform.lower()
    try:
        if platform == "twitter":
            # Decrypt Twitter access token
            access_token = decrypt_secret(creds.twitter_access_token)
            return await _publish_twitter(post_content, access_token)
        elif platform == "linkedin":
            # Decrypt LinkedIn access token
            access_token = decrypt_secret(creds.linkedin_access_token)
            return await _publish_linkedin(post_content, access_token)
        elif platform == "discord":
            webhook_url = decrypt_secret(creds.discord_webhook_url) if creds.discord_webhook_url else None
            if not webhook_url:
                return {"status": "error", "message": "Discord webhook URL not configured."}
            return await _publish_discord(post_content, webhook_url)
        elif platform == "medium":
            medium_token = decrypt_secret(creds.medium_integration_token) if creds.medium_integration_token else None
            if not medium_token:
                return {"status": "error", "message": "Medium integration token not configured."}
            return await _publish_medium(post_content, medium_token)
        else:
            return {"status": "error", "message": f"Unsupported platform: {platform}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def _publish_twitter(text: str, access_token: str) -> dict:
    """
    Post to Twitter using OAuth 2.0 Bearer token.
    """
    try:
        # Tweepy with OAuth2 bearer token
        client = tweepy.Client(bearer_token=access_token)
        # Run sync tweepy call in a thread so the event loop is not blocked
        response = await asyncio.to_thread(client.create_tweet, text=text)
        tweet_id = response.data['id']
        return {
            "status": "success",
            "platform": "twitter",
            "post_id": tweet_id,
            "url": f"https://twitter.com/user/status/{tweet_id}",
            "message": "Successfully posted to Twitter!"
        }
    except Exception as e:
        return {"status": "error", "message": f"Twitter API error: {str(e)}"}

async def _publish_linkedin(text: str, access_token: str) -> dict:
    """
    Post to LinkedIn using OAuth 2.0 token.
    Steps:
    1. Get user URN from /userinfo endpoint (OpenID Connect).
    2. Post to /ugcPosts.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    async with httpx.AsyncClient() as client:
        # 1. Get user info (OpenID Connect) to retrieve 'sub' (URN)
        userinfo_resp = await client.get("https://api.linkedin.com/v2/userinfo", headers=headers)
        if userinfo_resp.status_code != 200:
            return {"status": "error", "message": f"LinkedIn userinfo failed: {userinfo_resp.text}"}
        userinfo = userinfo_resp.json()
        person_urn = userinfo.get("sub")
        if not person_urn:
            return {"status": "error", "message": "Could not retrieve LinkedIn URN (sub)."}

        # 2. Create post
        post_data = {
            "author": f"urn:li:person:{person_urn}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }
        post_resp = await client.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers=headers,
            json=post_data
        )
        if post_resp.status_code == 201:
            post_id = post_resp.headers.get("x-linkedin-id", "unknown")
            return {
                "status": "success",
                "platform": "linkedin",
                "post_id": post_id,
                "message": "Successfully posted to LinkedIn!"
            }
        else:
            return {"status": "error", "message": f"LinkedIn post failed: {post_resp.text}"}

async def _publish_discord(text: str, webhook_url: str) -> dict:
    """
    Post to a Discord channel using a Webhook URL.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json={"content": text})
            if resp.status_code in (200, 204):
                return {
                    "status": "success",
                    "platform": "discord",
                    "message": "Successfully posted to Discord!"
                }
            else:
                return {"status": "error", "message": f"Discord post failed: {resp.status_code} - {resp.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Discord API error: {str(e)}"}

async def _publish_medium(text: str, token: str) -> dict:
    """
    Post to Medium using an Integration Token.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Charset": "utf-8"
    }
    
    # 1. We must extract a simple Title from the generated text
    # Assuming first line/sentence is the title
    title = text.split('\n')[0][:100] # Use first line, max 100 chars
    if not title:
        title = "Voice-To-Post Generated Output"
        
    try:
        async with httpx.AsyncClient() as client:
            # 1. Get user identity to fetch 'authorId'
            me_resp = await client.get("https://api.medium.com/v1/me", headers=headers)
            if me_resp.status_code != 200:
                return {"status": "error", "message": f"Medium identity fetch failed: {me_resp.text}"}
            me_data = me_resp.json()
            author_id = me_data.get("data", {}).get("id")
            
            if not author_id:
                return {"status": "error", "message": "Could not extract Medium author ID."}
            
            # 2. Create the post
            post_data = {
                "title": title,
                "contentFormat": "markdown",
                "content": text,
                "publishStatus": "public" # Could be draft too
            }
            
            post_resp = await client.post(
                f"https://api.medium.com/v1/users/{author_id}/posts",
                headers=headers,
                json=post_data
            )
            
            if post_resp.status_code == 201:
                resp_json = post_resp.json()
                post_url = resp_json.get("data", {}).get("url", "")
                return {
                    "status": "success",
                    "platform": "medium",
                    "url": post_url,
                    "message": "Successfully posted to Medium!"
                }
            else:
                return {"status": "error", "message": f"Medium post failed: {post_resp.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Medium API error: {str(e)}"}