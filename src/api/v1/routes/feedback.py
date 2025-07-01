"""
API endpoints for feedback on ATT&CK technique extraction (v1).
Implements the human feedback loop described in the technical documentation.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.middleware.auth import get_current_user
from src.api.middleware.rate_limit import tiered_limit
from src.database.postgresql import get_db
from src.enhanced_attack_extractor import get_extractor

# Create router
router = APIRouter(tags=["feedback"])


# Models
class HighlightedSegment(BaseModel):
    """Model for text segment highlighting"""

    text: str = Field(..., description="Highlighted text segment")
    start: int = Field(..., description="Start offset in original text")
    end: int = Field(..., description="End offset in original text")


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
    highlighted_segments: Optional[List[HighlightedSegment]] = Field(
        None, description="Text segments justifying technique attribution"
    )


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""

    status: str
    retraining_queued: bool = False
    feedback_id: Optional[int] = None


class FeedbackEntry(BaseModel):
    """Model for a feedback entry"""

    id: int
    technique_id: str
    feedback_type: str
    suggested_alternative: Optional[str] = None
    confidence_level: Optional[int] = None
    justification: Optional[str] = None
    created_at: str
    highlighted_segments: Optional[List[HighlightedSegment]] = None


class ModelTrainingRequest(BaseModel):
    """Model for requesting model retraining"""

    force: bool = Field(False, description="Force retraining even if not enough data")


# Endpoints
@router.post("/feedback", response_model=FeedbackResponse)
@tiered_limit(
    basic_limit="50/minute", premium_limit="100/minute", enterprise_limit="200/minute"
)
async def submit_feedback(
    request: Request,
    feedback: FeedbackSubmission,
    background_tasks: BackgroundTasks,
    user: Dict = Depends(get_current_user),
):
    """
    Submit feedback on technique extraction

    This endpoint allows analysts to provide feedback on extracted techniques, including
    corrections, confidence ratings, and highlighting text segments that justify attribution.
    This feedback is used to improve the extraction models over time.
    """
    # Get extractor instance
    extractor = get_extractor()

    # Process feedback
    success = extractor.process_feedback(
        analysis_id=feedback.analysis_id,
        technique_id=feedback.technique_id,
        feedback_type=feedback.feedback_type,
        user_id=user.get("user_id", "unknown"),
        suggested_technique_id=feedback.suggested_technique_id,
        confidence_level=feedback.confidence_level,
        justification_text=feedback.justification,
        highlighted_segments=(
            [dict(segment) for segment in feedback.highlighted_segments]
            if feedback.highlighted_segments
            else None
        ),
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to process feedback")

    # Check if we have enough feedback data to trigger retraining
    db = get_db()
    feedback_count = db.query_one(
        """
        SELECT COUNT(*) as count FROM analysis_feedback 
        WHERE created_at > (
            SELECT COALESCE(MAX(last_training_date), '1970-01-01'::timestamp) 
            FROM model_training_logs
        )
        """
    )

    retraining_queued = False

    if feedback_count and feedback_count.get("count", 0) >= 50:
        # Queue model retraining task
        from src.queue.manager import AnalysisQueueManager

        queue_manager = AnalysisQueueManager()

        try:
            background_tasks.add_task(queue_manager.enqueue_model_retraining)
            retraining_queued = True
        except Exception as e:
            # Log error but continue - feedback is still saved
            import logging

            logging.error(f"Failed to queue model retraining: {str(e)}")

    # Get the feedback ID
    feedback_id = db.query_one(
        """
        SELECT id FROM analysis_feedback
        WHERE analysis_id = %s AND technique_id = %s AND user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (feedback.analysis_id, feedback.technique_id, user.get("user_id")),
    )

    return {
        "status": "success",
        "retraining_queued": retraining_queued,
        "feedback_id": feedback_id.get("id") if feedback_id else None,
    }


@router.get("/feedback/{analysis_id}", response_model=List[FeedbackEntry])
async def get_feedback(
    request: Request, analysis_id: str, user: Dict = Depends(get_current_user)
):
    """
    Get feedback for an analysis

    This endpoint retrieves all feedback entries for a specific analysis job,
    including highlighted text segments.
    """
    # Get database connection
    db = get_db()

    # Verify the user has access to this analysis
    job = db.query_one(
        "SELECT user_id FROM analysis_jobs WHERE id = %s", (analysis_id,)
    )

    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Check permissions
    is_admin = user.get("is_admin", False)
    user_id = user.get("user_id", "unknown")

    if job.get("user_id") != user_id and not is_admin:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this analysis"
        )

    # Get feedback entries
    feedback_entries = db.query(
        """
        SELECT id, technique_id, feedback_type, suggested_alternative, 
               confidence_level, justification, created_at
        FROM analysis_feedback 
        WHERE analysis_id = %s
        ORDER BY created_at DESC
        """,
        (analysis_id,),
    )

    # Get highlighted segments for each feedback entry
    for entry in feedback_entries:
        if "id" in entry:
            highlights = db.query(
                """
                SELECT segment_text as text, start_offset as start, end_offset as end
                FROM feedback_highlights
                WHERE feedback_id = %s
                ORDER BY start_offset
                """,
                (entry["id"],),
            )
            entry["highlighted_segments"] = highlights or []

    return feedback_entries


@router.post("/feedback/retrain", response_model=Dict[str, Any])
@tiered_limit(basic_limit="1/day", premium_limit="5/day", enterprise_limit="10/day")
async def trigger_model_retraining(
    request: Request,
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    user: Dict = Depends(get_current_user),
):
    """
    Trigger model retraining based on collected feedback

    This endpoint allows administrators to manually trigger the model retraining process,
    which incorporates feedback data to improve extraction accuracy.
    """
    # Check admin permissions
    is_admin = user.get("is_admin", False)

    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin permission required")

    # Get database connection
    db = get_db()

    # Check if we have enough feedback data
    if not training_request.force:
        feedback_count = db.query_one(
            """
            SELECT COUNT(*) as count FROM analysis_feedback 
            WHERE created_at > (
                SELECT COALESCE(MAX(last_training_date), '1970-01-01'::timestamp) 
                FROM model_training_logs WHERE status = 'completed'
            )
            """
        )

        if feedback_count and feedback_count.get("count", 0) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Only {feedback_count.get('count', 0)} feedback entries since last training. Use force=true to override.",
                "feedback_count": feedback_count.get("count", 0),
            }

    # Queue model retraining task
    from src.queue.manager import AnalysisQueueManager

    queue_manager = AnalysisQueueManager()

    try:
        job = queue_manager.enqueue_model_retraining(forced=training_request.force)
        return {
            "status": "retraining_queued",
            "job_id": job.id,
            "message": "Model retraining has been queued",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to queue retraining: {str(e)}"
        )
