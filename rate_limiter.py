from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Create rate limiter instance
limiter = Limiter(key_func=get_remote_address)

# Rate limit configurations for different endpoint types
RATE_LIMITS = {
    "auth": "10/minute",           # Login/signup: 10 per minute
    "generation": "5/minute",      # Post generation: 5 per minute (resource intensive)
    "publish": "20/minute",        # Publishing: 20 per minute
    "upload": "10/minute",         # File uploads: 10 per minute
    "general": "60/minute",        # General API calls: 60 per minute
    "analytics": "30/minute",      # Analytics queries: 30 per minute
}

# Per-user limits (when authenticated)
USER_RATE_LIMITS = {
    "generation": "50/hour",       # 50 generations per hour per user
    "publish": "100/hour",         # 100 posts per hour per user
    "upload": "20/hour",           # 20 uploads per hour per user
}

def get_rate_limit_error_response():
    """Custom error response for rate limit exceeded."""
    return {
        "error": "rate_limit_exceeded",
        "message": "Too many requests. Please slow down and try again later.",
        "retry_after": "60 seconds"
    }
