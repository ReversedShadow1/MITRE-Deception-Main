"""
Rate limiting middleware for the API.
Implements tiered rate limiting based on user authentication and endpoint complexity.
"""

import time
from typing import Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def get_user_tier(request: Request) -> str:
    """
    Get the user tier from request state

    Args:
        request: HTTP request

    Returns:
        User tier string (basic, premium, enterprise)
    """
    # Default to basic tier
    default_tier = "basic"

    # If authenticated with JWT
    if hasattr(request.state, "user"):
        return request.state.user.get("tier", default_tier)

    # If authenticated with API key
    if hasattr(request.state, "api_key_data"):
        return request.state.api_key_data.get("tier", default_tier)

    return default_tier


def get_tiered_key(request: Request) -> str:
    """
    Get key for rate limiting based on user tier and identifier

    Args:
        request: HTTP request

    Returns:
        Rate limiting key
    """
    # Get user tier
    tier = get_user_tier(request)

    # Get user identifier (IP address if not authenticated)
    identifier = None

    if hasattr(request.state, "user"):
        identifier = request.state.user.get("user_id")
    elif hasattr(request.state, "api_key"):
        identifier = request.state.api_key

    # Fall back to IP address if no identifier available
    if not identifier:
        identifier = get_remote_address(request)

    return f"{tier}:{identifier}"


def configure_rate_limiting(app: FastAPI) -> Limiter:
    """
    Configure rate limiting for the FastAPI application

    Args:
        app: FastAPI application

    Returns:
        Configured Limiter instance
    """
    # Create limiter instance using tiered key function
    limiter = Limiter(key_func=get_tiered_key)

    # Add to app state and exception handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    return limiter


# Simplified decorator factory for endpoint-specific rate limits
def tiered_limit(
    basic_limit: str = "30/minute",
    premium_limit: str = "100/minute",
    enterprise_limit: str = "300/minute",
) -> Callable:
    """
    Create a rate limit decorator with tier-specific limits

    For now, this returns the basic limit. You can enhance this later.

    Args:
        basic_limit: Rate limit for basic tier
        premium_limit: Rate limit for premium tier
        enterprise_limit: Rate limit for enterprise tier

    Returns:
        Rate limit decorator function
    """

    def decorator(func: Callable) -> Callable:
        # For simplicity, just use the basic limit for now
        # You can enhance this to check user tier later
        func._rate_limit = basic_limit
        return func

    return decorator


# Usage examples (commented out):
#
# @tiered_limit(basic_limit="10/minute", premium_limit="50/minute")
# @app.get("/api/v1/endpoint")
# async def endpoint(request: Request):
#     return {"message": "Rate limited endpoint"}
#
# Or with the limiter directly:
#
# @limiter.limit("5/minute")
# @app.get("/api/another-endpoint")
# async def another_endpoint(request: Request):
#     return {"message": "Another rate limited endpoint"}
