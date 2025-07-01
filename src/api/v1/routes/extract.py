"""
API endpoints for ATT&CK technique extraction (v1).
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.middleware.auth import get_current_user
from src.api.middleware.rate_limit import tiered_limit
from src.enhanced_attack_extractor import get_extractor

# Create router
router = APIRouter(tags=["extraction"])


# Models
class TextInput(BaseModel):
    """Input model for technique extraction from text"""

    text: str = Field(..., description="Text to analyze for ATT&CK techniques")
    extractors: Optional[List[str]] = Field(
        None, description="List of extractors to use"
    )
    threshold: Optional[float] = Field(
        0.2, description="Confidence threshold for techniques"
    )
    top_k: Optional[int] = Field(
        10, description="Maximum number of techniques to return"
    )
    use_ensemble: Optional[bool] = Field(
        True, description="Whether to use ensemble method"
    )


class BatchTextInput(BaseModel):
    """Input model for batch technique extraction"""

    texts: List[str] = Field(..., description="List of texts to analyze")
    extractors: Optional[List[str]] = Field(
        None, description="List of extractors to use"
    )
    threshold: Optional[float] = Field(
        0.2, description="Confidence threshold for techniques"
    )
    top_k: Optional[int] = Field(
        10, description="Maximum number of techniques to return"
    )
    use_ensemble: Optional[bool] = Field(
        True, description="Whether to use ensemble method"
    )
    batch_size: Optional[int] = Field(5, description="Size of batches for processing")
    max_workers: Optional[int] = Field(
        2, description="Maximum number of parallel workers"
    )


class TechniqueResult(BaseModel):
    """Model for a technique extraction result"""

    technique_id: str
    confidence: float
    method: str
    matched_keywords: Optional[List[str]] = None
    cve_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Response model for technique extraction"""

    techniques: List[TechniqueResult]
    meta: Dict[str, Any]


# Endpoints
@router.post("/extract", response_model=ExtractionResponse)
@tiered_limit(
    basic_limit="30/minute", premium_limit="100/minute", enterprise_limit="300/minute"
)
async def extract_techniques(
    request: Request, input_data: TextInput, user: Dict = Depends(get_current_user)
):
    """
    Extract ATT&CK techniques from text

    This endpoint analyzes the provided text and identifies relevant MITRE ATT&CK techniques
    using multiple extraction methods.
    """
    # Get metrics manager from app state if it exists
    metrics_manager = getattr(request.app.state, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="api", status="started")

    extractor = get_extractor()

    try:
        # Record start time
        start_time = time.time()

        # Extract techniques
        results = extractor.extract_techniques(
            text=input_data.text,
            extractors=input_data.extractors,
            threshold=input_data.threshold,
            top_k=input_data.top_k,
            use_ensemble=input_data.use_ensemble,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track metrics if available
        if metrics_manager and results and "techniques" in results:
            technique_count = len(results["techniques"])
            metrics_manager.track_extraction(
                method="api", status="success", techniques_count=technique_count
            )

            for technique in results["techniques"]:
                if "confidence" in technique and "method" in technique:
                    metrics_manager.track_technique_confidence(
                        extractor_type=technique["method"],
                        confidence=technique["confidence"],
                    )

            # Add processing time to metrics
            metrics_manager.track_extraction_time(
                extractor_type="combined", duration=processing_time
            )

        # Attach user info to results metadata
        if "meta" in results:
            results["meta"]["user_id"] = user.get("user_id")
            results["meta"]["user_tier"] = user.get("tier", "basic")

        return results

    except Exception as e:
        # Track error if metrics manager available
        if metrics_manager:
            metrics_manager.track_extraction(method="api", status="error")

        # Log the error
        import logging

        logging.error(f"Error processing extraction request: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@router.post("/extract/batch", response_model=List[ExtractionResponse])
@tiered_limit(
    basic_limit="5/minute", premium_limit="15/minute", enterprise_limit="50/minute"
)
async def extract_techniques_batch(
    request: Request, input_data: BatchTextInput, user: Dict = Depends(get_current_user)
):
    """
    Extract ATT&CK techniques from multiple texts in batch

    This endpoint analyzes multiple texts in batch mode and identifies relevant MITRE ATT&CK
    techniques for each text.
    """
    # Track batch extraction request
    metrics_manager = getattr(request.app.state, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="batch_api", status="started")

    # Check if batch size is allowed for user tier
    tier = user.get("tier", "basic")
    max_batch_size = {"basic": 10, "premium": 50, "enterprise": 200}.get(tier, 10)

    if len(input_data.texts) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"{tier.capitalize()} tier allows maximum batch size of {max_batch_size}",
        )

    extractor = get_extractor()

    try:
        # Record start time
        start_time = time.time()

        # Process batch
        results = extractor.extract_techniques_batch(
            texts=input_data.texts,
            extractors=input_data.extractors,
            threshold=input_data.threshold,
            top_k=input_data.top_k,
            use_ensemble=input_data.use_ensemble,
            batch_size=input_data.batch_size,
            max_workers=input_data.max_workers,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track metrics if available
        if metrics_manager:
            metrics_manager.track_extraction(
                method="batch_api", status="success", techniques_count=len(results)
            )
            metrics_manager.track_extraction_time(
                extractor_type="batch", duration=processing_time
            )

        return results

    except Exception as e:
        # Track error if metrics manager available
        if metrics_manager:
            metrics_manager.track_extraction(method="batch_api", status="error")

        # Log the error
        import logging

        logging.error(f"Error processing batch extraction: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Batch extraction error: {str(e)}")


class DetailedExtractionResponse(BaseModel):
    """Response model for detailed technique extraction"""

    techniques: List[TechniqueResult]
    individual_extractor_results: Dict[str, List[TechniqueResult]]
    extraction_steps: List[Dict[str, Any]]
    meta: Dict[str, Any]


# Add this new endpoint to the router
@router.post("/extract/detailed", response_model=DetailedExtractionResponse)
@tiered_limit(
    basic_limit="20/minute", premium_limit="60/minute", enterprise_limit="120/minute"
)
async def extract_techniques_detailed(
    request: Request, input_data: TextInput, user: Dict = Depends(get_current_user)
):
    """
    Extract ATT&CK techniques from text with detailed steps and intermediate results

    This endpoint provides a detailed view of the extraction process, including:
    - Individual results from each extractor
    - Step-by-step tracking of the entire process
    - Final combined and ranked techniques
    """
    # Get metrics manager from app state
    metrics_manager = getattr(request.app, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="detailed_api", status="started")

    extractor = get_extractor()

    try:
        # Use the detailed extraction method
        results = extractor.extract_techniques_with_details(
            text=input_data.text,
            extractors=input_data.extractors,
            threshold=input_data.threshold,
            top_k=input_data.top_k,
            use_ensemble=input_data.use_ensemble,
        )

        # Track metrics if available
        if metrics_manager and results and "techniques" in results:
            technique_count = len(results["techniques"])
            metrics_manager.track_extraction(
                method="detailed_api",
                status="success",
                techniques_count=technique_count,
            )

        # Attach user info to results metadata
        if "meta" in results:
            results["meta"]["user_id"] = user.get("user_id")
            results["meta"]["user_tier"] = user.get("tier", "basic")

        return results

    except Exception as e:
        # Track error
        if metrics_manager:
            metrics_manager.track_extraction(method="detailed_api", status="error")

        # Log the error
        import logging

        logging.error(
            f"Error processing detailed extraction request: {str(e)}", exc_info=True
        )

        # Raise HTTP exception
        raise HTTPException(
            status_code=500, detail=f"Detailed extraction error: {str(e)}"
        )


# Modify the existing extract_techniques endpoint to use the detailed extraction method
@router.post("/extract", response_model=ExtractionResponse)
@tiered_limit(
    basic_limit="30/minute", premium_limit="100/minute", enterprise_limit="300/minute"
)
async def extract_techniques(
    request: Request, input_data: TextInput, user: Dict = Depends(get_current_user)
):
    """
    Extract ATT&CK techniques from text

    This endpoint analyzes the provided text and identifies relevant MITRE ATT&CK techniques
    using multiple extraction methods.
    """
    # Get metrics manager from app state
    metrics_manager = getattr(request.app, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="api", status="started")

    extractor = get_extractor()

    try:
        # Use the detailed extraction method
        detailed_results = extractor.extract_techniques_with_details(
            text=input_data.text,
            extractors=input_data.extractors,
            threshold=input_data.threshold,
            top_k=input_data.top_k,
            use_ensemble=input_data.use_ensemble,
        )

        # Convert to standard response format (backward compatibility)
        results = {
            "techniques": detailed_results["techniques"],
            "meta": detailed_results["meta"],
        }

        # Track metrics if available
        if metrics_manager and results and "techniques" in results:
            technique_count = len(results["techniques"])
            metrics_manager.track_extraction(
                method="api", status="success", techniques_count=technique_count
            )

        # Attach user info to results metadata
        if "meta" in results:
            results["meta"]["user_id"] = user.get("user_id")
            results["meta"]["user_tier"] = user.get("tier", "basic")

        return results

    except Exception as e:
        # Track error
        if metrics_manager:
            metrics_manager.track_extraction(method="api", status="error")

        # Log the error
        import logging

        logging.error(f"Error processing extraction request: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
