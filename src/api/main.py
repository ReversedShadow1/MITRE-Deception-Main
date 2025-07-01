'''"""
Application entry point for the MITRE Deception Project API.
"""

import os

import uvicorn

# Import the FastAPI app
from src.api.app import app

# Run the application if this script is executed directly
if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    workers = int(os.environ.get("WORKERS", "1"))

    # Run with uvicorn
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload, workers=workers)
'''


# src/main.py - Main application entry point

import logging
import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    # Import app module
    from src.api.app import app

    # Add optimizations
    _add_optimizations(app)

    return app


def _add_optimizations(app: FastAPI):
    """Add performance optimizations to the application"""

    # Initialize and add caching if enabled
    if os.environ.get("ENABLE_CACHING", "true").lower() == "true":
        try:
            from src.api.v2.routes.Optimized_Cache import CacheMiddleware, TieredCache

            # Get cache configuration from environment
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            memory_cache_size = int(os.environ.get("CACHE_MEMORY_SIZE", "1000"))

            # Create cache
            cache = TieredCache(
                redis_url=redis_url,
                memory_cache_size=memory_cache_size,
                file_cache_dir="cache",
                enable_memory_cache=True,
                enable_redis_cache=bool(redis_url),
                enable_file_cache=True,
            )

            # Add middleware
            app.add_middleware(CacheMiddleware, cache=cache)
            logger.info("Added caching middleware")

            # Store cache in app state for later access
            app.state.cache = cache
        except Exception as e:
            logger.error(f"Failed to initialize caching: {e}")

    # Use connection pooling if enabled
    if os.environ.get("USE_CONNECTION_POOLING", "true").lower() == "true":
        try:
            import src.database.connection_pool

            # Override get_db function
            from src.database import postgresql

            postgresql.get_db = src.database.connection_pool.get_pooled_db
            logger.info("Enabled database connection pooling")
        except Exception as e:
            logger.error(f"Failed to enable connection pooling: {e}")


# Main function to start the app directly (for development)
if __name__ == "__main__":
    import uvicorn

    # Create the app
    app = create_app()

    # Run the server
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )
