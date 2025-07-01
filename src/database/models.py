"""
Database models for PostgreSQL and Neo4j integration.
Contains Pydantic models for request/response validation and database interactions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

# ---- Request/Response Models ----


class AnalysisRequest(BaseModel):
    """Request model for text analysis"""

    text: str
    input_type: str = "text"  # text, file, url
    extractors: Optional[List[str]] = ["rule_based", "bm25", "ner", "semantic", "kev"]
    threshold: float = 0.2
    top_k: int = 10
    use_ensemble: bool = True
    user_id: Optional[UUID] = None

    class Config:
        extra = "allow"


class TechniqueResult(BaseModel):
    """Model for a single technique extraction result"""

    technique_id: str
    technique_name: Optional[str] = None
    confidence: float
    method: str
    matched_keywords: Optional[List[str]] = None
    matched_entities: Optional[List[str]] = None
    cve_id: Optional[str] = None
    tactics: Optional[List[str]] = None
    description: Optional[str] = None
    url: Optional[str] = None

    class Config:
        orm_mode = True


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""

    job_id: UUID
    status: str  # pending, processing, completed, failed
    techniques: Optional[List[TechniqueResult]] = None
    estimated_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None

    class Config:
        orm_mode = True


class TechniqueVisualization(BaseModel):
    """Model for MITRE Navigator visualization"""

    techniqueID: str
    score: float
    color: str
    comment: Optional[str] = None
    enabled: bool = True
    metadata: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class UserStats(BaseModel):
    """Model for user statistics"""

    user_id: UUID
    username: str
    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    bookmarked_analyses: int
    unique_techniques_found: int
    avg_confidence: float
    linked_layers: int

    class Config:
        orm_mode = True


class DashboardMetrics(BaseModel):
    """Model for dashboard metrics"""

    analysis_count: int
    top_techniques: List[Dict[str, Any]]
    api_usage: Dict[str, Any]
    recent_analyses: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class Neo4jQueryRequest(BaseModel):
    """Model for Neo4j query requests"""

    query: str
    parameters: Optional[Dict[str, Any]] = {}


# ---- Database Models ----


class AnalysisJob(BaseModel):
    """Database model for analysis jobs"""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    name: Optional[str] = None
    status: str
    input_type: str
    input_data: str
    extractors_used: List[str]
    threshold: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None

    class Config:
        orm_mode = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insert"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "name": self.name,
            "status": self.status,
            "input_type": self.input_type,
            "input_data": self.input_data,
            "extractors_used": self.extractors_used,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "processing_time_ms": self.processing_time_ms,
        }


class AnalysisResult(BaseModel):
    """Database model for analysis results"""

    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    technique_id: str
    technique_name: Optional[str] = None
    confidence: float
    method: str
    matched_keywords: Optional[List[str]] = None
    cve_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insert"""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "confidence": self.confidence,
            "method": self.method,
            "matched_keywords": self.matched_keywords,
            "cve_id": self.cve_id,
            "created_at": self.created_at,
        }


class UserMetric(BaseModel):
    """Database model for user metrics"""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    metric_type: str
    metric_value: Dict[str, Any]
    time_period: Optional[str] = None
    date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True


class ApiUsage(BaseModel):
    """Database model for API usage tracking"""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    endpoint: str
    method: str
    status_code: int
    response_time_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True


class AnalysisBookmark(BaseModel):
    """Database model for analysis bookmarks"""

    id: int = None
    user_id: UUID
    job_id: UUID
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True


# ---- Neo4j Models ----


class Neo4jTechnique(BaseModel):
    """Model for Neo4j ATT&CK technique node"""

    technique_id: str
    name: str
    description: Optional[str] = None
    is_subtechnique: bool = False
    parent_technique_id: Optional[str] = None
    url: Optional[str] = None
    tactics: Optional[List[str]] = None

    class Config:
        extra = "allow"


class Neo4jJobNode(BaseModel):
    """Model for Neo4j analysis job node"""

    job_id: str
    user_id: str
    status: str
    input_type: str
    extractors_used: List[str]
    threshold: float
    created_at: str
    completed_at: Optional[str] = None
    processing_time_ms: Optional[int] = None

    class Config:
        extra = "allow"


class Neo4jResultNode(BaseModel):
    """Model for Neo4j analysis result node"""

    result_id: str
    confidence: float
    method: str
    matched_keywords: Optional[List[str]] = None
    cve_id: Optional[str] = None
    created_at: str

    class Config:
        extra = "allow"


# ---- Utility functions ----


def convert_technique_results(
    neo4j_results: List[Dict[str, Any]],
) -> List[TechniqueResult]:
    """Convert Neo4j results to TechniqueResult models"""
    results = []

    for record in neo4j_results:
        # Extract technique properties
        technique = record.get("technique", {})

        result = TechniqueResult(
            technique_id=technique.get("technique_id", ""),
            technique_name=technique.get("name", ""),
            confidence=record.get("confidence", 0.0),
            method=record.get("method", "unknown"),
            matched_keywords=record.get("matched_keywords", []),
            cve_id=record.get("cve_id"),
            tactics=record.get("tactics", []),
            description=technique.get("description", ""),
            url=technique.get("url", ""),
        )

        results.append(result)

    return results
