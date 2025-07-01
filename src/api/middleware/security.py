"""
Security middleware for the API.
Implements security headers and CSRF protection.
"""

import os
import secrets
import time
from typing import Callable, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for setting security headers"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Add security headers to response

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response with security headers
        """
        response = await call_next(request)

        # Content Security Policy
        # Only allow resources from same origin by default
        response.headers["Content-Security-Policy"] = self._build_csp_header()

        # Prevent browsers from performing MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Frame options
        response.headers["X-Frame-Options"] = "DENY"

        # HSTS (only in production)
        if os.environ.get("ENVIRONMENT", "development") == "production":
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains"

        return response

    def _build_csp_header(self) -> str:
        """
        Build Content Security Policy header

        Returns:
            CSP header string
        """
        # Define CSP directives
        directives = {
            "default-src": ["'self'"],
            "script-src": ["'self'"],
            "style-src": ["'self'"],
            "img-src": ["'self'", "data:"],
            "font-src": ["'self'"],
            "connect-src": ["'self'"],
            "frame-src": ["'none'"],
            "object-src": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
            "frame-ancestors": ["'none'"],
            "upgrade-insecure-requests": [],
        }

        # Add additional sources based on environment
        if os.environ.get("ENVIRONMENT") == "development":
            # Allow inline scripts for development only
            directives["script-src"].append("'unsafe-inline'")

        # Build CSP header string
        csp_parts = []
        for directive, sources in directives.items():
            if sources:
                csp_parts.append(f"{directive} {' '.join(sources)}")
            else:
                csp_parts.append(directive)

        return "; ".join(csp_parts)


# CSRF Protection
class CSRFProtection:
    """CSRF protection utilities"""

    # Cache of valid tokens
    _tokens: Dict[str, float] = {}

    # Token cleanup interval in seconds
    _cleanup_interval = 3600

    # Last cleanup timestamp
    _last_cleanup = time.time()

    @classmethod
    def generate_token(cls) -> str:
        """
        Generate a new CSRF token

        Returns:
            CSRF token string
        """
        token = secrets.token_urlsafe(32)
        cls._tokens[token] = time.time() + 86400  # Valid for 24 hours

        # Cleanup expired tokens periodically
        cls._cleanup_tokens()

        return token

    @classmethod
    def validate_token(cls, token: str) -> bool:
        """
        Validate a CSRF token

        Args:
            token: CSRF token to validate

        Returns:
            Whether token is valid
        """
        if token not in cls._tokens:
            return False

        # Check if token is expired
        if time.time() > cls._tokens[token]:
            del cls._tokens[token]
            return False

        return True

    @classmethod
    def _cleanup_tokens(cls) -> None:
        """Clean up expired tokens"""
        now = time.time()

        # Only clean up at most once per cleanup interval
        if now - cls._last_cleanup < cls._cleanup_interval:
            return

        # Remove expired tokens
        expired_tokens = [
            token for token, expiry in cls._tokens.items() if now > expiry
        ]
        for token in expired_tokens:
            del cls._tokens[token]

        cls._last_cleanup = now


async def get_csrf_token(request: Request) -> Optional[str]:
    """
    Get CSRF token from request

    Args:
        request: HTTP request

    Returns:
        CSRF token if present, None otherwise
    """
    # Try to get from header first
    token = request.headers.get("X-CSRF-Token")

    # Fall back to form data
    if not token and request.method in ["POST", "PUT", "DELETE", "PATCH"]:
        try:
            form_data = await request.form()
            token = form_data.get("csrf_token")
        except:
            pass

    return token


def csrf_protection(excludes: Set[str] = None):
    """
    Dependency for CSRF protection

    Args:
        excludes: Set of excluded path prefixes

    Returns:
        Dependency function
    """
    excludes = excludes or set()

    async def _csrf_protect(request: Request):
        # Skip CSRF check for excluded paths
        path = request.url.path
        if any(path.startswith(prefix) for prefix in excludes):
            return

        # Skip CSRF check for safe methods
        if request.method in ["GET", "HEAD", "OPTIONS", "TRACE"]:
            return

        # Validate CSRF token
        token = await get_csrf_token(request)
        if not token or not CSRFProtection.validate_token(token):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")

    return _csrf_protect


def configure_csrf_protection(app: FastAPI, exclude_paths: List[str] = None) -> None:
    """
    Configure CSRF protection for the FastAPI application

    Args:
        app: FastAPI application
        exclude_paths: List of paths to exclude from CSRF protection
    """
    exclude_paths = set(exclude_paths or ["/api/v1/health", "/metrics"])

    # Add CSRF token endpoint
    @app.get("/api/csrf-token")
    async def get_csrf():
        token = CSRFProtection.generate_token()
        return {"csrf_token": token}

    # Add CSRF protection dependency to app
    app.dependency_overrides[csrf_protection] = csrf_protection(exclude_paths)
