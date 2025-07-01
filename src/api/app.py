"""
Main FastAPI application for the MITRE Deception Project API.
"""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

# Import middleware
from src.api.middleware.auth import APIKeyAuthMiddleware, JWTAuthMiddleware
from src.api.middleware.rate_limit import configure_rate_limiting
from src.api.middleware.security import (
    SecurityHeadersMiddleware,
    configure_csrf_protection,
)

# Import monitoring setup
from src.monitoring.setup import setup_monitoring, wrap_database_connections

# Create FastAPI app
app = FastAPI(
    title="MITRE Deception Project API",
    description="API for extracting MITRE ATT&CK techniques from threat intelligence and generating deception strategies",
    version="1.0.0",
)


# Setup monitoring
monitoring_components = setup_monitoring(
    app=app,
    service_name="attack_extractor",
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    log_dir=os.environ.get("LOG_DIR", "logs"),
    elasticsearch_host=os.environ.get("ELASTICSEARCH_HOST"),
    metrics_port=int(os.environ.get("METRICS_PORT", "8000")),
    prometheus_push_gateway=os.environ.get("PROMETHEUS_PUSH_GATEWAY"),
)

# Wrap database connections with monitoring
wrap_database_connections(monitoring_components)

# Keep track of the metrics manager for later use
metrics_manager = monitoring_components["metrics_manager"]

# Configure rate limiting
limiter = configure_rate_limiting(app)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    JWTAuthMiddleware, exclude_paths=["/api/v1/health", "/api/v1/extract", "/metrics"]
)
app.add_middleware(APIKeyAuthMiddleware, exclude_paths=["/api/v1/health", "/metrics"])

# Configure CSRF protection
configure_csrf_protection(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.v1.routes import extract as extract_v1
from src.api.v1.routes import feedback as feedback_v1
from src.api.v1.routes import health as health_v1
from src.api.v2.routes import extract as extract_v2

# Include routers
app.include_router(health_v1.router, prefix="/api/v1")
app.include_router(extract_v1.router, prefix="/api/v1")
app.include_router(feedback_v1.router, prefix="/api/v1")
app.include_router(extract_v2.router, prefix="/api/v2")


# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for API requests"""
    # Log the exception
    import logging

    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    # Return a standardized error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
            if os.environ.get("ENVIRONMENT") == "development"
            else "An unexpected error occurred",
            "request_id": request.headers.get("X-Request-ID", "unknown"),
        },
    )


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema for the API"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="MITRE Deception Project API",
        version="1.0.0",
        description="API for extracting MITRE ATT&CK techniques and generating deception strategies",
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"] = {
        "securitySchemes": {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "BearerAuth": {"type": "http", "scheme": "bearer"},
        }
    }

    # Add global security requirement
    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    # Add example requests/responses
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "examples" not in openapi_schema["components"]:
        openapi_schema["components"]["examples"] = {}

    # Add extraction examples
    openapi_schema["components"]["examples"]["BasicExtractionRequest"] = {
        "summary": "Basic extraction request",
        "value": {
            "text": "The attacker used PowerShell to execute malicious commands and exfiltrated data using DNS tunneling. They established persistence through a scheduled task and disabled Windows Defender to avoid detection.",
            "extractors": ["rule_based", "bm25", "kev"],
            "threshold": 0.2,
            "top_k": 10,
            "use_ensemble": True,
        },
    }

    openapi_schema["components"]["examples"]["SuccessfulExtractionResponse"] = {
        "summary": "Successful extraction response",
        "value": {
            "techniques": [
                {
                    "technique_id": "T1059.001",
                    "confidence": 0.89,
                    "method": "rule_based",
                    "matched_keywords": ["powershell", "command execution"],
                    "name": "PowerShell",
                    "description": "Adversaries may abuse PowerShell commands and scripts for execution...",
                },
                {
                    "technique_id": "T1048.003",
                    "confidence": 0.76,
                    "method": "bm25",
                    "matched_keywords": ["dns tunneling", "dns exfiltration"],
                    "name": "Exfiltration Over Alternative Protocol: DNS",
                },
                {
                    "technique_id": "T1053.005",
                    "confidence": 0.72,
                    "method": "rule_based",
                    "matched_keywords": ["scheduled task", "persistence"],
                    "name": "Scheduled Task",
                },
                {
                    "technique_id": "T1562.001",
                    "confidence": 0.65,
                    "method": "ensemble",
                    "matched_keywords": ["disabled defender", "disable security tools"],
                    "name": "Disable or Modify Tools",
                },
            ],
            "meta": {
                "text_length": 195,
                "processing_time": 0.842,
                "extractors_used": {"rule_based": True, "bm25": True, "kev": True},
                "ensemble_used": True,
                "threshold": 0.2,
                "technique_count": 4,
                "using_neo4j": True,
                "user_tier": "premium",
            },
        },
    }

    # Add enhanced extraction examples
    openapi_schema["components"]["examples"]["EnhancedExtractionRequest"] = {
        "summary": "Enhanced extraction request with additional options",
        "value": {
            "text": "The attacker exploited CVE-2021-44228 (Log4Shell) to gain initial access, then used PowerShell to execute commands and exfiltrate data. They disabled Windows Defender and established persistence through registry modifications.",
            "extractors": ["rule_based", "bm25", "kev", "ner", "semantic"],
            "threshold": 0.2,
            "top_k": 10,
            "use_ensemble": True,
            "include_context": True,
            "include_relationships": True,
            "return_navigator_layer": True,
            "content_type": "text",
        },
    }

    openapi_schema["components"]["examples"]["EnhancedExtractionResponse"] = {
        "summary": "Enhanced extraction response with context and Navigator layer",
        "value": {
            "techniques": [
                {
                    "technique_id": "T1190",
                    "confidence": 0.95,
                    "method": "kev",
                    "cve_id": "CVE-2021-44228",
                    "name": "Exploit Public-Facing Application",
                    "description": "Adversaries may attempt to exploit a weakness in...",
                    "context": {
                        "tactics": ["initial-access"],
                        "platforms": ["Linux", "Windows", "macOS", "SaaS"],
                        "data_sources": ["Application Log", "Network Traffic"],
                        "mitigations": [{"id": "M1051", "name": "Update Software"}],
                        "similar_techniques": [
                            {
                                "technique_id": "T1133",
                                "name": "External Remote Services",
                                "relationship_type": "SIMILAR_TO",
                            }
                        ],
                    },
                    "relationships": [
                        {
                            "relationship_type": "ENABLES",
                            "related_type": "AttackTechnique",
                            "related_id": "T1059.001",
                            "related_name": "PowerShell",
                        },
                        {
                            "relationship_type": "RELATED_TO",
                            "related_type": "CVE",
                            "related_id": "CVE-2021-44228",
                            "related_name": "Log4Shell",
                        },
                    ],
                },
                {
                    "technique_id": "T1059.001",
                    "confidence": 0.87,
                    "method": "rule_based",
                    "matched_keywords": ["powershell", "command execution"],
                    "name": "PowerShell",
                    "context": {
                        "tactics": ["execution"],
                        "platforms": ["Windows"],
                        "data_sources": [
                            "Command Execution",
                            "Process Monitoring",
                            "Windows Event Logs",
                        ],
                        "mitigations": [
                            {
                                "id": "M1042",
                                "name": "Disable or Remove Feature or Program",
                            }
                        ],
                        "similar_techniques": [],
                    },
                },
            ],
            "meta": {
                "text_length": 238,
                "processing_time": 1.24,
                "extractors_used": {
                    "rule_based": True,
                    "bm25": True,
                    "kev": True,
                    "ner": True,
                    "semantic": True,
                },
                "ensemble_used": True,
                "content_type": "text",
                "user_tier": "enterprise",
            },
            "navigator_layer": {
                "name": "Extraction Results",
                "versions": {"attack": "13", "navigator": "4.8.0", "layer": "4.4"},
                "domain": "enterprise-attack",
                "description": "Layer generated on 2025-05-20 14:30:45",
                "techniques": [
                    {
                        "techniqueID": "T1190",
                        "score": 0.95,
                        "color": "#ff6666",
                        "comment": "Extracted using kev method",
                        "enabled": True,
                        "metadata": [
                            {"name": "method", "value": "kev"},
                            {"name": "extraction_confidence", "value": "95%"},
                        ],
                    },
                    {
                        "techniqueID": "T1059.001",
                        "score": 0.87,
                        "color": "#ff6666",
                        "comment": "Extracted using rule_based method",
                        "enabled": True,
                        "metadata": [
                            {"name": "method", "value": "rule_based"},
                            {"name": "extraction_confidence", "value": "87%"},
                        ],
                    },
                ],
            },
        },
    }

    # Add document upload examples
    openapi_schema["components"]["examples"]["DocumentUploadResponse"] = {
        "summary": "Response from document upload and analysis",
        "value": {
            "techniques": [
                {
                    "technique_id": "T1566.001",
                    "confidence": 0.92,
                    "method": "ensemble",
                    "matched_keywords": [
                        "phishing",
                        "malicious attachment",
                        "malicious email",
                    ],
                    "name": "Spear Phishing Attachment",
                },
                {
                    "technique_id": "T1204.002",
                    "confidence": 0.81,
                    "method": "semantic",
                    "name": "User Execution: Malicious File",
                },
            ],
            "meta": {
                "filename": "threat_report_2025.pdf",
                "content_type": "pdf",
                "file_size": 1250000,
                "processing_time": 2.34,
                "extractors_used": {
                    "rule_based": True,
                    "bm25": True,
                    "semantic": True,
                    "ner": True,
                },
                "user_tier": "premium",
            },
        },
    }

    # Add feedback examples
    openapi_schema["components"]["examples"]["FeedbackSubmissionRequest"] = {
        "summary": "Feedback submission for extracted technique",
        "value": {
            "analysis_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "technique_id": "T1059.001",
            "feedback_type": "correct",
            "confidence_level": 5,
            "justification": "The report clearly mentions PowerShell script execution for lateral movement.",
            "highlighted_segments": [
                {
                    "text": "attacker used PowerShell scripts to execute commands across the network",
                    "start": 120,
                    "end": 185,
                }
            ],
        },
    }

    openapi_schema["components"]["examples"]["FeedbackSubmissionCorrection"] = {
        "summary": "Feedback submission correcting extracted technique",
        "value": {
            "analysis_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "technique_id": "T1021.006",
            "feedback_type": "incorrect",
            "suggested_technique_id": "T1021.002",
            "confidence_level": 4,
            "justification": "The report describes SMB access, not Windows Remote Management.",
        },
    }

    openapi_schema["components"]["examples"]["FeedbackSubmissionResponse"] = {
        "summary": "Response to feedback submission",
        "value": {"status": "success", "retraining_queued": True, "feedback_id": 123},
    }

    # Add health check examples
    openapi_schema["components"]["examples"]["HealthResponse"] = {
        "summary": "Health check response",
        "value": {
            "status": "healthy",
            "version": "1.0.0",
            "extractors": [
                {"name": "rule_based", "loaded": True, "available": True},
                {"name": "bm25", "loaded": True, "available": True},
                {"name": "ner", "loaded": True, "available": True},
                {"name": "semantic", "loaded": True, "available": True},
                {"name": "classifier", "loaded": True, "available": True},
                {"name": "kev", "loaded": True, "available": True},
            ],
            "databases": [
                {
                    "type": "neo4j",
                    "connected": True,
                    "version": "Neo4j 5.0.0",
                    "latency_ms": 12.5,
                },
                {
                    "type": "postgresql",
                    "connected": True,
                    "version": "PostgreSQL 14.5",
                    "latency_ms": 8.3,
                },
                {
                    "type": "redis",
                    "connected": True,
                    "version": "7.0.4",
                    "latency_ms": 3.1,
                },
            ],
            "environment": "production",
            "gpu_available": True,
            "uptime_seconds": 345600,
        },
    }

    # Add navigator layer examples
    openapi_schema["components"]["examples"]["NavigatorLayerRequest"] = {
        "summary": "Request to generate Navigator layer",
        "value": {
            "technique_ids": ["T1566.001", "T1059.001", "T1053.005", "T1190"],
            "scores": {
                "T1566.001": 0.92,
                "T1059.001": 0.85,
                "T1053.005": 0.76,
                "T1190": 0.65,
            },
            "name": "APT123 Campaign Analysis",
            "description": "Techniques observed in APT123 campaign during Q1 2025",
            "include_context": True,
        },
    }

    # Add error examples
    openapi_schema["components"]["examples"]["AuthenticationError"] = {
        "summary": "Authentication error response",
        "value": {
            "error": "Authentication required",
            "detail": "API key or JWT token required",
        },
    }

    openapi_schema["components"]["examples"]["ValidationError"] = {
        "summary": "Validation error response",
        "value": {
            "error": "Validation error",
            "detail": [
                {
                    "loc": ["body", "text"],
                    "msg": "field required",
                    "type": "value_error.missing",
                }
            ],
        },
    }

    openapi_schema["components"]["examples"]["RateLimitError"] = {
        "summary": "Rate limit exceeded error",
        "value": {
            "error": "Rate limit exceeded",
            "detail": "Request rate limit of 30/minute exceeded. Try again in 45 seconds.",
        },
    }

    # Attach examples to specific endpoints (this would be more comprehensive in actual implementation)
    # For now, we'll add some path-level examples to demonstrate the approach
    if "paths" in openapi_schema:
        # Add examples to extraction endpoint
        if "/api/v1/extract" in openapi_schema["paths"]:
            if "post" in openapi_schema["paths"]["/api/v1/extract"]:
                openapi_schema["paths"]["/api/v1/extract"]["post"]["requestBody"] = {
                    "content": {
                        "application/json": {
                            "examples": {
                                "BasicExtractionRequest": {
                                    "$ref": "#/components/examples/BasicExtractionRequest"
                                }
                            }
                        }
                    }
                }
                openapi_schema["paths"]["/api/v1/extract"]["post"]["responses"] = {
                    "200": {
                        "description": "Successful extraction",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "SuccessfulResponse": {
                                        "$ref": "#/components/examples/SuccessfulExtractionResponse"
                                    }
                                }
                            }
                        },
                    },
                    "401": {
                        "description": "Authentication error",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "AuthError": {
                                        "$ref": "#/components/examples/AuthenticationError"
                                    }
                                }
                            }
                        },
                    },
                    "429": {
                        "description": "Rate limit exceeded",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "RateLimit": {
                                        "$ref": "#/components/examples/RateLimitError"
                                    }
                                }
                            }
                        },
                    },
                }

        # Add examples to v2 enhanced extraction endpoint
        if "/api/v2/extract" in openapi_schema["paths"]:
            if "post" in openapi_schema["paths"]["/api/v2/extract"]:
                openapi_schema["paths"]["/api/v2/extract"]["post"]["requestBody"] = {
                    "content": {
                        "application/json": {
                            "examples": {
                                "EnhancedExtractionRequest": {
                                    "$ref": "#/components/examples/EnhancedExtractionRequest"
                                }
                            }
                        }
                    }
                }
                openapi_schema["paths"]["/api/v2/extract"]["post"]["responses"] = {
                    "200": {
                        "description": "Successful enhanced extraction",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "EnhancedResponse": {
                                        "$ref": "#/components/examples/EnhancedExtractionResponse"
                                    }
                                }
                            }
                        },
                    }
                }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
