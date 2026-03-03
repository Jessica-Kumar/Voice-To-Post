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
        response = client.create_tweet(text=text)
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