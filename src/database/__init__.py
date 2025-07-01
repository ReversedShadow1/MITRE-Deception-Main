"""# src/database/__init__.py
from .neo4j import get_neo4j
from .postgresql import get_db

__all__ = ["get_db", "get_neo4j"]"""


from .neo4j import get_neo4j
from .postgresql import get_db

# Add connection pool
try:
    from .connection_pool import get_pooled_db

    __all__ = ["get_db", "get_neo4j", "get_pooled_db"]
except ImportError:
    __all__ = ["get_db", "get_neo4j"]
