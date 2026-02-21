import os
import httpx

# Read Deepgram API key from environment
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY must be set in the environment.")

async def transcribe_audio_bytes(audio_bytes: bytes, content_type: str = "audio/wav") -> str:
    """
    Asynchronously transcribes audio bytes using the Deepgram REST API via httpx.
    
    Args:
        audio_bytes (bytes): The audio file content in bytes.
        content_type (str): The mimetype of the audio file.
        
    Returns:
        str: The transcribed text.
    """
    # Deepgram API endpoint with nova-3 and smart_format
    url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type
    }
    
    try:
        # Use httpx async client for making the REST HTTP call
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                content=audio_bytes,
                timeout=60.0 # Generous timeout in case of long audio
            )
            
        response.raise_for_status()
        
        # Parse the JSON response following Deepgram's return structure
        data = response.json()
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        return transcript
        
    except httpx.HTTPStatusError as e:
        print(f"Deepgram HTTP error: {e.response.status_code} - {e.response.text}")
        return f"Error transcribing audio: Deepgram API returned {e.response.status_code}"
    except Exception as e:
        print(f"Deepgram transcription processing error: {e}")
        return f"Error transcribing audio: {str(e)}"
