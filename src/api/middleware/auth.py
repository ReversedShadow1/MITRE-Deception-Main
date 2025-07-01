"""
Authentication middleware for the API.
Implements JWT authentication and API key verification.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import jwt
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# Configure logging
logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "development_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 60 * 60  # 1 hour in seconds

# API Key Configuration
API_KEYS = {}  # In production, these would be loaded from a secure store


def load_api_keys():
    """Load API keys from environment or database"""
    # For development/testing, load from environment variable
    test_api_key = os.environ.get("TEST_API_KEY")
    test_user_uuid = os.environ.get(
        "TEST_USER_UUID", "00000000-0000-0000-0000-000000000000"
    )
    if test_api_key:
        API_KEYS[test_api_key] = {
            "user_id": test_user_uuid,
            "rate_limit": 100,  # requests per minute
            "tier": "premium",
        }

    # TODO: In production, load from database
    # This is just a placeholder for the actual implementation
    pass


load_api_keys()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a new JWT token

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time

    Returns:
        JWT token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(seconds=JWT_EXPIRATION))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict:
    """
    Decode and validate a JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token data

    Raises:
        jwt.InvalidTokenError: If token is invalid
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def verify_api_key(api_key: str) -> Optional[Dict]:
    """
    Verify an API key

    Args:
        api_key: API key to verify

    Returns:
        API key metadata if valid, None otherwise
    """
    return API_KEYS.get(api_key)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication"""

    def __init__(self, app, exclude_paths: List[str] = None):
        """
        Initialize the middleware

        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from authentication
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []
        self.bearer = HTTPBearer(auto_error=False)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        # Skip authentication for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Try to get the JWT token
        credentials: Optional[HTTPAuthorizationCredentials] = await self.bearer(request)

        if not credentials:
            # No JWT token, but might use API key instead
            return await call_next(request)

        # Verify the JWT token
        try:
            token_data = decode_token(credentials.credentials)
            # Attach user data to request state
            request.state.user = token_data
            return await call_next(request)
        except jwt.InvalidTokenError as e:
            # Log the error
            logger.warning(f"Invalid JWT token: {str(e)}")
            return Response(
                status_code=401,
                content={"error": "Invalid authentication token", "detail": str(e)},
                media_type="application/json",
            )


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication"""

    def __init__(self, app, exclude_paths: List[str] = None):
        """
        Initialize the middleware

        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from authentication
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        # Skip authentication for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # If user is already authenticated via JWT, continue
        if hasattr(request.state, "user"):
            return await call_next(request)

        # Check for API key
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return Response(
                status_code=401,
                content={
                    "error": "Authentication required",
                    "detail": "API key or JWT token required",
                },
                media_type="application/json",
            )

        # Verify the API key
        api_key_data = verify_api_key(api_key)

        if not api_key_data:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return Response(
                status_code=401,
                content={"error": "Invalid API key"},
                media_type="application/json",
            )

        # Attach API key data to request state
        request.state.api_key = api_key
        request.state.api_key_data = api_key_data

        # Also set user data for consistency
        request.state.user = {
            "user_id": api_key_data["user_id"],
            "auth_type": "api_key",
            "tier": api_key_data["tier"],
        }

        return await call_next(request)


def get_current_user(request: Request) -> Dict:
    """
    Get the current authenticated user

    Args:
        request: HTTP request

    Returns:
        User data

    Raises:
        HTTPException: If user is not authenticated
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Authentication required")

    return request.state.user
