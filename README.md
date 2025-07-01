# MITRE ATT&CK Technique Extractor

A high-performance API for extracting MITRE ATT&CK techniques from threat intelligence and generating deception strategies.

## Overview

This project provides a sophisticated API for analyzing security-related text and extracting relevant MITRE ATT&CK techniques. It utilizes multiple extraction methods, including rule-based, BM25, semantic similarity, and machine learning approaches to deliver accurate technique identification.

Key features include:
- Multi-method technique extraction with ensemble capabilities
- Integration with MITRE ATT&CK, CISA KEV, CAPEC, and CWE frameworks
- Neo4j graph database for advanced relationship mapping
- High-performance API with caching, rate limiting, and load balancing
- Comprehensive security measures including API key authentication and CSRF protection
- Detailed performance metrics and monitoring
- Background processing for resource-intensive tasks

## Architecture

The system is designed with a microservices architecture:

- **API Server**: FastAPI-based API for handling requests
- **Worker Nodes**: Background processing for intensive workloads
- **Neo4j**: Graph database for storing MITRE ATT&CK data and relationships
- **PostgreSQL**: Relational database for API operation data
- **Redis**: Caching and message queuing
- **RabbitMQ**: Advanced message queuing for distributed tasks (optional integration)

![deploy2](https://github.com/user-attachments/assets/80918fc6-8654-4216-92cf-baf20d9bd2dd)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- PostgreSQL 14+
- Neo4j 5+

- Redis 7+

### Installation with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/attack-extractor.git
   cd attack-extractor
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

### Manual Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download required models:
   ```bash
   python download_models.py
   ```

3. Initialize databases:
   ```bash
   # PostgreSQL migrations
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/v001_base_schema.sql
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/v002_analysis_tables.sql
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/v003_feedback_enhancements.sql
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/v004_detailed_metrics.sql
   ```

4. Start the application:
   ```bash
   python -m src.api.main
   ```

## API Usage

### Authentication

API requests require authentication using either an API key or JWT token:

```python
import requests

# Using API key
headers = {"X-API-Key": "your-api-key"}
response = requests.post("http://localhost:8000/api/v2/extract/text", 
                         json={"text": "The malware used PowerShell to download additional payloads."}, 
                         headers=headers)
```

### Basic Technique Extraction

```python
# Extract techniques from text
response = requests.post(
    "http://localhost:8000/api/v2/extract/text",
    json={
        "text": "The threat actors used spear-phishing emails with malicious attachments.",
        "threshold": 0.2,
        "top_k": 10,
        "include_context": True
    },
    headers=headers
)
```

### Advanced Options

```python
# Using ensemble method with specific extractors
response = requests.post(
    "http://localhost:8000/api/v2/extract/text",
    json={
        "text": "The attack involved lateral movement using stolen credentials.",
        "extractors": ["rule_based", "bm25", "semantic"],
        "use_ensemble": True,
        "include_relationships": True,
        "return_navigator_layer": True
    },
    headers=headers
)
```

### Curl commands

**Cheking System Health**

```bash
curl -X GET "http://localhost:8000/api/v1/health" \
  -H "X-API-Key: test_api_key_for_development"
```

*API_V1* extract endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_api_key_for_development" \
  -d '{
    "text": "Indicators showed DLL side-loading and data compression which led to data loss.",
    "extractors": ["rule_based", "bm25", "kev", "semantic", "ner"],
    "threshold": 0.1,
    "top_k": 15
  }'
```

*API_V1* detailed extract endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/extract/detailed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_api_key_for_development" \
  -d '{
    "text": "The attackers used PowerShell to execute malicious code and performed lateral movement. They also used phishing emails with malicious attachments containing CVE-2021-44228 exploits.",
    "extractors": ["rule_based", "bm25", "kev", "semantic", "ner"],
    "threshold": 0.1,
    "top_k": 15
  }'
```

*API_V2* extract endpoint:

```bash
curl -X POST "http://localhost:8000/api/v2/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_api_key_for_development" \
  -d '{
    "text": "The attacker used PowerShell to execute malicious commands and exfiltrated data using DNS tunneling. T1548",
    "extractors": ["keyword", "entity", "bert"],
    "threshold": 0.3,
    "top_k": 5,
    "use_ensemble": true,
    "include_context": true,
    "include_relationships": true,
    "return_navigator_layer": true,
    "content_type": "text"
  }'
```

*API_V2* stream extraction endpoint:

```bash
curl -X POST "http://localhost:8000/api/v2/extract/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_api_key_for_development" \
  -d '{
    "text": "The attacker used PowerShell to execute malicious commands and exfiltrated data using DNS tunneling.",
    "show_progress": true
  }'
```

*API_V2* feedback submission endpoint:

```bash
curl -X POST "http://localhost:8000/api/v2/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_api_key_for_development" \
  -d '{
    "job_id": "JOB_ID",
    "technique_id": "T1059.001",
    "feedback_type": "correct",
    "confidence_level": 5,
    "justification": "PowerShell was correctly identified as a technique"
  }'
```

*API_V2* checking job metrics endpoint:

```bash
curl -X GET "http://localhost:8000/api/v2/jobs/JOB_ID/metrics" \
  -H "X-API-Key: test_api_key_for_development"
```

*API_V2* document upload endpoint (supporting pdf/html/markdown/txt):

```bash
curl -X POST "http://localhost:8000/api/v2/upload-document" \
  -H "X-API-Key: test_api_key_for_development" \
  -F "file=@document.pdf" \
  -F "threshold=0.1" \
  -F "top_k=10" \
  -F "use_ensemble=true" \
  -F "include_context=true" \
  -F "include_relationships=true"
```

*API_V2* MITRE Navigator Layer Generation endpoint:

```bash
curl -X POST "http://localhost:8000/api/v2/navigator/layer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_api_key_for_development" \
  -d '{
    "techniques": ["T1059.001", "T1048.003"],
    "name": "Test Layer",
    "description": "Test layer with manually specified techniques"
  }'
```

Cashe Statistics **(Admin Only)**

```bash
curl -X GET "http://localhost:8000/api/v2/cache/stats" \
  -H "X-API-Key: test_api_key_for_development"
```

Cashe Cleanup **(Admin Only)**

```bash
curl -X POST "http://localhost:8000/api/v2/cache/cleanup" \
  -H "X-API-Key: test_api_key_for_development"
```

## Development

### Project Structure

```
├── src/                  # Source code
│   ├── api/              # API implementation
│   ├── extractors/       # Technique extraction methods
│   ├── preprocessing/    # Text preprocessing
│   ├── monitoring/       # Metrics and monitoring
│   ├── queue/            # Background task processing
│   └── database/         # Database connections
├── migrations/           # Database migrations
├── models/               # ML model storage
├── data/                 # Data files
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile.worker     # Docker Worker
└── Dockerfile            # Docker configuration
```

### CI/CD Pipeline

The project uses GitHub Actions for CI/CD with the following stages:
1. Code Quality (Black, isort, flake8, mypy)
2. Unit Tests
3. Integration Tests
4. Security Scanning
5. Docker Image Building
6. Deployment

## Security

This project implements several security measures:
- API key and JWT authentication
- Rate limiting to prevent abuse
- CSRF protection
- Input validation
- Docker image vulnerability scanning
- Dependency security checks


## Acknowledgements

- MITRE ATT&CK® is a registered trademark of The MITRE Corporation
- This project utilizes various open-source libraries and frameworks
