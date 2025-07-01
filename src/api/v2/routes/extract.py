"""
Enhanced API endpoints for ATT&CK technique extraction (v2).
Includes advanced extraction, batch processing, streaming, and visualization.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.middleware.auth import get_current_user
from src.api.middleware.rate_limit import tiered_limit

# Import enhanced extraction manager
from src.api.v2.routes.Enhanced_Extraction_Manager import EnhancedExtractionManager

# Import authentication middleware
from src.database.postgresql import get_db

# Import text preprocessing
from src.preprocessing.text_processor import TextProcessor

# Create router
router = APIRouter(tags=["extraction"])


# Models
class EnhancedTextInput(BaseModel):
    """Enhanced input model for technique extraction from text"""

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
    include_context: Optional[bool] = Field(
        False, description="Include contextual information for techniques"
    )
    include_relationships: Optional[bool] = Field(
        False, description="Include technique relationships"
    )
    return_navigator_layer: Optional[bool] = Field(
        False, description="Return MITRE Navigator layer data"
    )
    content_type: Optional[str] = Field(
        "text", description="Content type (text, html, markdown)"
    )
    use_cache: Optional[bool] = Field(True, description="Whether to use result cache")
    custom_preprocessor: Optional[Dict[str, Any]] = Field(
        None, description="Custom preprocessing configuration"
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
    include_context: Optional[bool] = Field(
        False, description="Include contextual information for techniques"
    )
    include_relationships: Optional[bool] = Field(
        False, description="Include technique relationships"
    )
    return_navigator_layer: Optional[bool] = Field(
        False, description="Return MITRE Navigator layer data"
    )
    batch_size: Optional[int] = Field(5, description="Size of batches for processing")
    use_cache: Optional[bool] = Field(True, description="Whether to use result cache")


class TechniqueContext(BaseModel):
    """Context information for a technique"""

    tactics: List[str] = Field(..., description="ATT&CK tactics")
    platforms: Optional[List[str]] = Field(None, description="Affected platforms")
    data_sources: Optional[List[str]] = Field(
        None, description="Data sources for detection"
    )
    mitigations: Optional[List[Dict[str, str]]] = Field(
        None, description="Related mitigations"
    )
    similar_techniques: Optional[List[Dict[str, Any]]] = Field(
        None, description="Similar techniques"
    )


class EnhancedTechniqueResult(BaseModel):
    """Enhanced model for a technique extraction result"""

    technique_id: str
    confidence: float
    method: str
    matched_keywords: Optional[List[str]] = None
    matched_entities: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    cve_id: Optional[str] = None
    cve_ids: Optional[List[str]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    context: Optional[TechniqueContext] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    component_scores: Optional[Dict[str, float]] = None
    extractors_used: Optional[List[str]] = None


class EnhancedExtractionResponse(BaseModel):
    """Enhanced response model for technique extraction"""

    techniques: List[EnhancedTechniqueResult]
    meta: Dict[str, Any]
    navigator_layer: Optional[Dict[str, Any]] = None


class StreamingExtractionRequest(BaseModel):
    """Request model for streaming extraction"""

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
    show_progress: Optional[bool] = Field(
        True, description="Whether to show extraction progress"
    )


class NavigatorLayerRequest(BaseModel):
    """Request model for Navigator layer generation"""

    technique_ids: List[str] = Field(..., description="List of technique IDs")
    scores: Optional[Dict[str, float]] = Field(
        None, description="Technique scores (ID -> score)"
    )
    name: Optional[str] = Field("Generated Layer", description="Layer name")
    description: Optional[str] = Field(None, description="Layer description")
    include_context: Optional[bool] = Field(
        False, description="Include contextual information"
    )


class FeedbackSubmission(BaseModel):
    """Model for feedback submission"""

    analysis_id: str = Field(..., description="ID of the analysis job")
    technique_id: str = Field(..., description="Technique ID receiving feedback")
    feedback_type: str = Field(
        ..., description="Type of feedback ('correct', 'incorrect', 'unsure')"
    )
    suggested_technique_id: Optional[str] = Field(
        None, description="Alternative technique suggested by analyst"
    )
    confidence_level: Optional[int] = Field(
        None, description="Analyst confidence rating (1-5)"
    )
    justification: Optional[str] = Field(
        None, description="Text explaining the feedback decision"
    )
    extractors_used: Optional[List[str]] = Field(
        None, description="Extractors that were used for this result"
    )


# Singleton manager instance
_extraction_manager = None


def get_extraction_manager():
    """Get or create the extraction manager instance"""
    global _extraction_manager

    if _extraction_manager is None:
        # Load technique data
        from src.enhanced_attack_extractor import get_extractor

        extractor = get_extractor()

        if not extractor:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize extractor. Check server logs for details.",
            )

        # Create manager
        _extraction_manager = EnhancedExtractionManager(
            techniques_data=extractor.techniques,
            technique_keywords=extractor.technique_keywords,
            use_optimized_extractors=True,
            use_caching=True,
            use_async=True,
            cache_type="memory",  # Can be 'memory', 'file', or 'redis'
            cache_dir="cache/extraction",
            redis_url=None,  # Set to Redis URL if available
            use_neo4j=True,
            max_workers=4,
        )

    return _extraction_manager


# Endpoints
@router.post("/extract", response_model=EnhancedExtractionResponse)
@tiered_limit(
    basic_limit="30/minute", premium_limit="100/minute", enterprise_limit="300/minute"
)
async def extract_techniques_enhanced(
    request: Request,
    input_data: EnhancedTextInput,
    user: Dict = Depends(get_current_user),
):
    """
    Enhanced extraction of ATT&CK techniques from text

    This endpoint provides an enhanced version of technique extraction with additional options:
    - Contextual information for techniques
    - Technique relationships
    - MITRE Navigator layer generation
    - Advanced text preprocessing
    """
    # Get metrics manager from app state if it exists
    metrics_manager = getattr(request.app.state, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="api_v2", status="started")

    # Get extraction manager
    manager = get_extraction_manager()

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    # Get user tier and ID
    user_id = user.get("user_id", "anonymous")
    tier = user.get("tier", "basic")

    # Check rate limits
    is_allowed, limit_info = manager.check_request_limits(
        user_id=user_id, tier=tier, text_length=len(input_data.text)
    )

    if not is_allowed:
        if metrics_manager:
            metrics_manager.track_extraction(method="api_v2", status="rate_limited")

        # Return rate limit error
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": limit_info},
        )

    try:
        # Preprocess text if needed
        if input_data.content_type != "text" or input_data.custom_preprocessor:
            # Initialize text processor
            text_processor = TextProcessor()

            # Apply custom preprocessing config if provided
            if input_data.custom_preprocessor:
                # Apply custom config (implementation depends on TextProcessor capabilities)
                pass

            # Preprocess text
            processed_text = text_processor.preprocess(
                text=input_data.text, content_type=input_data.content_type
            )
        else:
            processed_text = input_data.text

        # Record start time
        start_time = time.time()

        # Extract techniques
        results = manager.extract_techniques(
            text=processed_text,
            extractors=input_data.extractors,
            threshold=input_data.threshold,
            top_k=input_data.top_k,
            use_ensemble=input_data.use_ensemble,
            include_context=input_data.include_context,
            include_relationships=input_data.include_relationships,
            return_navigator_layer=input_data.return_navigator_layer,
            user_id=user_id,
            tier=tier,
            request_id=request_id,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track metrics
        if metrics_manager:
            techniques = results.get("techniques", [])
            metrics_manager.track_extraction(
                method="api_v2", status="success", techniques_count=len(techniques)
            )
            metrics_manager.track_extraction_time(
                extractor_type="enhanced", duration=processing_time
            )

        # Add user info to results metadata
        if "meta" in results:
            results["meta"]["user_id"] = user_id
            results["meta"]["user_tier"] = tier

        return results

    except Exception as e:
        # Track error
        if metrics_manager:
            metrics_manager.track_extraction(method="api_v2", status="error")

        # Log error
        import logging

        logging.error(f"Error in enhanced extraction: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(
            status_code=500, detail=f"Enhanced extraction error: {str(e)}"
        )


@router.post("/extract/batch", response_model=List[EnhancedExtractionResponse])
@tiered_limit(
    basic_limit="5/minute", premium_limit="15/minute", enterprise_limit="50/minute"
)
async def extract_techniques_batch(
    request: Request, input_data: BatchTextInput, user: Dict = Depends(get_current_user)
):
    """
    Extract ATT&CK techniques from multiple texts in batch

    This endpoint analyzes multiple texts in batch mode and identifies relevant MITRE ATT&CK
    techniques for each text with enhanced features.
    """
    # Track batch extraction request
    metrics_manager = getattr(request.app.state, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="batch_api_v2", status="started")

    # Get extraction manager
    manager = get_extraction_manager()

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    # Get user tier and ID
    user_id = user.get("user_id", "anonymous")
    tier = user.get("tier", "basic")

    # Check batch size limit
    is_allowed, limit_info = manager.check_request_limits(
        user_id=user_id,
        tier=tier,
        text_length=sum(len(text) for text in input_data.texts),
        batch_size=len(input_data.texts),
    )

    if not is_allowed:
        if metrics_manager:
            metrics_manager.track_extraction(
                method="batch_api_v2", status="rate_limited"
            )

        # Return rate limit error
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": limit_info},
        )

    try:
        # Record start time
        start_time = time.time()

        # Process batch
        results = manager.extract_techniques_batch(
            texts=input_data.texts,
            extractors=input_data.extractors,
            threshold=input_data.threshold,
            top_k=input_data.top_k,
            use_ensemble=input_data.use_ensemble,
            include_context=input_data.include_context,
            include_relationships=input_data.include_relationships,
            return_navigator_layer=input_data.return_navigator_layer,
            user_id=user_id,
            tier=tier,
            request_id=request_id,
            batch_size=input_data.batch_size,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track metrics
        if metrics_manager:
            metrics_manager.track_extraction(
                method="batch_api_v2",
                status="success",
                techniques_count=sum(len(r.get("techniques", [])) for r in results),
            )
            metrics_manager.track_extraction_time(
                extractor_type="batch_enhanced", duration=processing_time
            )

        return results

    except Exception as e:
        # Track error
        if metrics_manager:
            metrics_manager.track_extraction(method="batch_api_v2", status="error")

        # Log error
        import logging

        logging.error(f"Error in batch extraction: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Batch extraction error: {str(e)}")


@router.post("/extract/stream")
async def extract_techniques_streaming(
    request: Request,
    input_data: StreamingExtractionRequest,
    user: Dict = Depends(get_current_user),
):
    """
    Streaming extraction of ATT&CK techniques with real-time updates

    This endpoint streams extraction results as they become available, providing
    real-time updates for long-running extractions.
    """
    # Get user tier and ID
    user_id = user.get("user_id", "anonymous")
    tier = user.get("tier", "basic")

    # Check tier permissions - streaming only available for premium and enterprise
    '''if tier == "basic":
        return JSONResponse(
            status_code=403,
            content={
                "error": "Feature not available",
                "detail": "Streaming extraction requires premium or enterprise tier",
            },
        )'''

    # Get extraction manager
    manager = get_extraction_manager()

    # Check rate limits
    is_allowed, limit_info = manager.check_request_limits(
        user_id=user_id, tier=tier, text_length=len(input_data.text)
    )

    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": limit_info},
        )

    # Define streaming response generator
    async def streaming_extraction():
        # Start with progress indicator
        if input_data.show_progress:
            yield json.dumps({"status": "starting", "progress": 0.0}) + "\n"

        try:
            # Mock implementation of streaming for now
            # In a real implementation, this would use async extraction with callbacks

            # Extract with each extractor in sequence for streaming
            extractors = input_data.extractors or ["rule_based", "bm25", "ner", "kev"]
            all_results = {}

            for i, extractor_name in enumerate(extractors):
                # Skip if extractor not available
                if extractor_name not in manager.extractors:
                    continue

                # Update progress
                if input_data.show_progress:
                    progress = (i / len(extractors)) * 0.7  # 70% for extraction
                    yield json.dumps(
                        {
                            "status": "extracting",
                            "progress": progress,
                            "extractor": extractor_name,
                        }
                    ) + "\n"

                # Extract with this extractor
                extractor = manager.extractors[extractor_name]

                # Handle different parameter naming conventions
                if extractor_name in ["rule_based", "ner"]:
                    results = extractor.extract_techniques(
                        text=input_data.text,
                        min_confidence=input_data.threshold,
                        max_results=input_data.top_k,
                    )
                else:
                    results = extractor.extract_techniques(
                        text=input_data.text,
                        threshold=input_data.threshold,
                        top_k=input_data.top_k,
                    )

                # Store results
                all_results[extractor_name] = results

                # Stream partial results
                partial_results = {
                    "status": "partial_results",
                    "extractor": extractor_name,
                    "techniques": results,
                    "progress": (i + 1) / len(extractors) * 0.7,
                }

                yield json.dumps(partial_results) + "\n"

                # Small delay for demonstration
                await asyncio.sleep(0.5)

            # Update progress for ensemble
            if input_data.show_progress:
                yield json.dumps({"status": "ensembling", "progress": 0.8}) + "\n"

            # Use ensemble if requested
            if input_data.use_ensemble and all_results:
                # Use advanced ensemble
                techniques = manager.ensemble.ensemble_extractors(
                    text=input_data.text,
                    extractor_results=all_results,
                    threshold=input_data.threshold,
                    max_results=input_data.top_k,
                )
            else:
                # Combine results without ensemble
                all_technique_results = []
                for results in all_results.values():
                    all_technique_results.extend(results)

                # Remove duplicates and sort by confidence
                seen_techniques = set()
                techniques = []

                for result in sorted(
                    all_technique_results,
                    key=lambda x: x.get("confidence", 0),
                    reverse=True,
                ):
                    tech_id = result.get("technique_id")
                    if tech_id and tech_id not in seen_techniques:
                        seen_techniques.add(tech_id)
                        techniques.append(result)

                        if len(techniques) >= input_data.top_k:
                            break

            # Final results
            final_results = {
                "status": "completed",
                "techniques": techniques,
                "meta": {
                    "text_length": len(input_data.text),
                    "extractors_used": {extractor: True for extractor in extractors},
                    "ensemble_used": input_data.use_ensemble,
                    "threshold": input_data.threshold,
                    "technique_count": len(techniques),
                },
                "progress": 1.0,
            }

            yield json.dumps(final_results) + "\n"

        except Exception as e:
            # Error in extraction
            error_response = {"status": "error", "error": str(e), "progress": 1.0}

            yield json.dumps(error_response) + "\n"

    # Return streaming response
    return StreamingResponse(streaming_extraction(), media_type="application/x-ndjson")


@router.post("/upload-document", response_model=EnhancedExtractionResponse)
@tiered_limit(
    basic_limit="5/minute", premium_limit="20/minute", enterprise_limit="60/minute"
)
async def extract_from_document(
    request: Request,
    file: UploadFile = File(...),
    extractors: str = Form(None),
    threshold: float = Form(0.2),
    top_k: int = Form(10),
    use_ensemble: bool = Form(True),
    include_context: bool = Form(False),
    include_relationships: bool = Form(False),
    return_navigator_layer: bool = Form(False),
    user: Dict = Depends(get_current_user),
):
    """
    Extract ATT&CK techniques from uploaded document

    This endpoint accepts document uploads (PDF, HTML, plain text) and extracts
    techniques from the document content with enhanced features.
    """
    # Track extraction request
    metrics_manager = getattr(request.app.state, "metrics_manager", None)

    if metrics_manager:
        metrics_manager.track_extraction(method="document_upload_v2", status="started")

    # Get extraction manager
    manager = get_extraction_manager()

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    # Get user tier and ID
    user_id = user.get("user_id", "anonymous")
    tier = user.get("tier", "basic")

    try:
        # Read file content
        file_content = await file.read()

        # Check file size limit
        is_allowed, limit_info = manager.check_request_limits(
            user_id=user_id, tier=tier, text_length=len(file_content)
        )

        if not is_allowed:
            if metrics_manager:
                metrics_manager.track_extraction(
                    method="document_upload_v2", status="rate_limited"
                )

            # Return rate limit error
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "detail": limit_info},
            )

        # Determine content type from file extension
        filename = file.filename.lower()
        content_type = "text"

        if filename.endswith(".pdf"):
            content_type = "pdf"
        elif filename.endswith(".html") or filename.endswith(".htm"):
            content_type = "html"
        elif filename.endswith(".md") or filename.endswith(".markdown"):
            content_type = "markdown"

        # For PDF, we need to save to disk temporarily for processing
        temp_file_path = None
        if content_type == "pdf":
            import os
            import tempfile

            # Create temporary file
            fd, temp_file_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)

            # Write content to temporary file
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

        # Process the document
        text_processor = TextProcessor()

        if content_type == "pdf" and temp_file_path:
            # Extract text from PDF
            processed_text = text_processor.preprocess(
                text="", content_type="pdf", file_path=temp_file_path
            )

            # Clean up temporary file
            import os

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        else:
            # Process other content types
            processed_text = text_processor.preprocess(
                text=file_content.decode("utf-8", errors="replace"),
                content_type=content_type,
            )

        # Parse extractors string to list if provided
        extractors_list = None
        if extractors:
            extractors_list = [e.strip() for e in extractors.split(",")]

        # Record start time
        start_time = time.time()

        # Extract techniques
        results = manager.extract_techniques(
            text=processed_text,
            extractors=extractors_list,
            threshold=threshold,
            top_k=top_k,
            use_ensemble=use_ensemble,
            include_context=include_context,
            include_relationships=include_relationships,
            return_navigator_layer=return_navigator_layer,
            user_id=user_id,
            tier=tier,
            request_id=request_id,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track metrics
        if metrics_manager:
            techniques = results.get("techniques", [])
            metrics_manager.track_extraction(
                method="document_upload_v2",
                status="success",
                techniques_count=len(techniques),
            )
            metrics_manager.track_extraction_time(
                extractor_type="document_enhanced", duration=processing_time
            )

        # Add file info to metadata
        if "meta" in results:
            results["meta"]["filename"] = file.filename
            results["meta"]["content_type"] = content_type
            results["meta"]["user_tier"] = tier
            results["meta"]["file_size"] = len(file_content)

        return results

    except Exception as e:
        # Track error
        if metrics_manager:
            metrics_manager.track_extraction(
                method="document_upload_v2", status="error"
            )

        # Clean up temporary file if it exists
        import os

        if (
            "temp_file_path" in locals()
            and temp_file_path
            and os.path.exists(temp_file_path)
        ):
            os.remove(temp_file_path)

        # Log error
        import logging

        logging.error(f"Error in document extraction: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(
            status_code=500, detail=f"Document extraction error: {str(e)}"
        )


@router.post("/navigator/layer", response_model=Dict[str, Any])
@tiered_limit(
    basic_limit="10/minute", premium_limit="30/minute", enterprise_limit="100/minute"
)
async def generate_navigator_layer(
    request: Request,
    layer_request: NavigatorLayerRequest,
    user: Dict = Depends(get_current_user),
):
    """
    Generate MITRE Navigator layer from technique IDs

    This endpoint creates a MITRE Navigator-compatible layer file for visualizing
    techniques and their scores.
    """
    # Validate input
    if not layer_request.technique_ids:
        raise HTTPException(
            status_code=400, detail="At least one technique ID is required"
        )

    # Get extraction manager
    manager = get_extraction_manager()

    try:
        # Generate layer
        navigator_layer = manager._generate_navigator_layer(
            techniques=layer_request.technique_ids,
            scores=layer_request.scores,
            name=layer_request.name,
            description=layer_request.description,
        )

        # Add additional technique context if requested
        if layer_request.include_context and navigator_layer.get("techniques"):
            for technique in navigator_layer["techniques"]:
                tech_id = technique.get("techniqueID")
                if tech_id:
                    # Get technique data
                    tech_data = manager.techniques_data.get(tech_id, {})

                    # Add to metadata
                    if tech_data and "metadata" in technique:
                        technique["metadata"].extend(
                            [
                                {
                                    "name": "description",
                                    "value": tech_data.get("description", "")[:100]
                                    + "...",
                                },
                                {
                                    "name": "platforms",
                                    "value": ", ".join(tech_data.get("platforms", [])),
                                },
                            ]
                        )

        return navigator_layer

    except Exception as e:
        # Log error
        import logging

        logging.error(f"Error generating Navigator layer: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Layer generation error: {str(e)}")


@router.post("/feedback", response_model=Dict[str, Any])
@tiered_limit(
    basic_limit="50/minute", premium_limit="100/minute", enterprise_limit="200/minute"
)
async def submit_feedback(
    request: Request,
    feedback: FeedbackSubmission,
    user: Dict = Depends(get_current_user),
):
    """
    Submit feedback on technique extraction

    This endpoint allows analysts to provide feedback on extracted techniques, including
    corrections, confidence ratings, and suggestions for improvement.
    This feedback is used to improve the extraction models over time.
    """
    # Get extraction manager
    manager = get_extraction_manager()

    try:
        # Add user ID to feedback data
        feedback_data = feedback.dict()
        feedback_data["user_id"] = user.get("user_id", "anonymous")

        # Process feedback
        success = manager.process_feedback(feedback_data)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to process feedback")

        # Check if we should trigger retraining
        # This would be implemented in a more robust way in a real system

        return {"status": "success", "message": "Feedback submitted successfully"}

    except Exception as e:
        # Log error
        import logging

        logging.error(f"Error processing feedback: {str(e)}", exc_info=True)

        # Raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats(request: Request, user: Dict = Depends(get_current_user)):
    """
    Get cache statistics

    This endpoint provides statistics about the result cache, including size,
    hit rate, and memory usage.
    """
    # Only admin users can access cache stats
    if not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin permission required")

    # Get extraction manager
    manager = get_extraction_manager()

    # Get cache stats
    stats = manager.get_cache_stats()

    return stats


@router.post("/cache/cleanup", response_model=Dict[str, Any])
async def cleanup_cache(request: Request, user: Dict = Depends(get_current_user)):
    """
    Clean up expired cache entries

    This endpoint removes expired entries from the result cache.
    """
    # Only admin users can clean up cache
    if not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin permission required")

    # Get extraction manager
    manager = get_extraction_manager()

    # Clean up cache
    removed_count = manager.cleanup_cache()

    return {"status": "success", "removed_entries": removed_count}


# Add to src/api/v2/routes/extract.py


@router.get("/jobs/{job_id}/metrics", response_model=Dict[str, Any])
@tiered_limit(
    basic_limit="10/minute", premium_limit="30/minute", enterprise_limit="100/minute"
)
async def get_job_metrics(
    request: Request, job_id: str, user: Dict = Depends(get_current_user)
):
    """
    Get complete metrics for a job

    This endpoint retrieves all metrics, results, and artifacts for a specific analysis job.
    """
    # Check user authorization (user should only access their own jobs)
    job = get_db().query_one(
        """
        SELECT user_id FROM analysis_jobs WHERE id = %s
        """,
        (job_id,),
    )

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["user_id"] != user.get("user_id") and not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Not authorized to access this job")

    # Get complete metrics
    from src.database.metrics_recorder import get_complete_job_metrics

    metrics = get_complete_job_metrics(job_id)

    return metrics
