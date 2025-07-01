FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies including PostgreSQL client
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY migrations/ ./migrations/
#COPY scripts/ ./scripts/

# Copy download script
COPY download_models.py /app/download_models.py

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    NEO4J_URI=bolt://neo4j:7687 \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=password \
    NEO4J_DATABASE=pipe \
    POSTGRES_HOST=postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=postgres \
    POSTGRES_DB=attack_extractor \
    AUTO_DOWNLOAD_MODELS=true \
    INITIALIZE_DATABASE=true


ENV PYTHONPATH=/app

# Create startup script
# In your Dockerfile, replace the startup script creation with:
RUN echo '#!/bin/bash\n\
\n\
echo "Starting ATT&CK Extractor..."\n\
\n\
# Wait for Neo4j to be ready\n\
echo "Waiting for Neo4j..."\n\
until curl -s http://neo4j:7474 > /dev/null; do\n\
    echo -n "."\n\
    sleep 2\n\
done\n\
echo " Neo4j is ready!"\n\
\n\
# Wait for PostgreSQL to be ready\n\
echo "Waiting for PostgreSQL..."\n\
until pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER; do\n\
    echo -n "."\n\
    sleep 2\n\
done\n\
echo " PostgreSQL is ready!"\n\
\n\
# Initialize databases if needed BEFORE starting the app\n\
if [ "$INITIALIZE_DATABASE" = "true" ]; then\n\
    echo "Initializing PostgreSQL database..."\n\
    # Run migrations in order\n\
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f /app/migrations/v001_base_schema.sql || echo "v001 migration failed or already applied"\n\
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f /app/migrations/v002_analysis_tables.sql || echo "v002 migration failed or already applied"\n\
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f /app/migrations/v003_feedback_enhancements.sql || echo "v003 migration failed or already applied"\n\
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f /app/migrations/v004_detailed_metrics.sql || echo "v004 migration failed or already applied"\n\
    echo "PostgreSQL initialization complete."\n\
    \n\
    echo "Initializing Neo4j database..."\n\
    # Use timeout to prevent hanging, and handle errors gracefully\n\
    timeout 600 python -c "\n\
try:\n\
    from src.mitre_integration import IntegrationManager\n\
    import os\n\
    manager = IntegrationManager(os.environ.get('"'"'NEO4J_URI'"'"'), os.environ.get('"'"'NEO4J_USER'"'"'), os.environ.get('"'"'NEO4J_PASSWORD'"'"'))\n\
    manager.run_full_update()\n\
    manager.close()\n\
    print('"'"'Neo4j initialization completed successfully'"'"')\n\
except Exception as e:\n\
    print(f'"'"'Neo4j initialization failed: {e}'"'"')\n\
    print('"'"'Application will continue with local data fallback'"'"')\n\
" || echo "Neo4j initialization failed, application will use local data fallback"\n\
    echo "Database initialization complete."\n\
fi\n\
\n\
# Check if models need to be downloaded\n\
if [ ! -f "/app/models/.download_complete" ] && [ "$AUTO_DOWNLOAD_MODELS" = "true" ]; then\n\
    echo "Downloading models..."\n\
    python /app/download_models.py || echo "Model download failed, will download on demand"\n\
fi\n\
\n\
# Start the application\n\
echo "Starting application..."\n\
exec python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000' > /app/startup.sh \
    && chmod +x /app/startup.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start application
CMD ["/app/startup.sh"]