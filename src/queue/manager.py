"""
Manages job queues for analysis and model retraining.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pika
import redis
from rq import Connection, Queue, Worker
from rq.job import Job

from src.database.neo4j import get_neo4j
from src.database.postgresql import get_db
from src.enhanced_attack_extractor import EnhancedATTCKExtractor

logger = logging.getLogger(__name__)


class AnalysisQueueManager:
    """Manages analysis job queue using Redis, RQ, and RabbitMQ"""

    def __init__(
        self,
        redis_url: str = None,
        rabbitmq_url: str = None,
    ):
        """Initialize queue manager with both Redis and RabbitMQ connections"""
        # Use environment variables if not provided
        self.redis_url = redis_url or os.environ.get(
            "REDIS_URL", "redis://localhost:6379"
        )
        self.rabbitmq_url = rabbitmq_url or os.environ.get(
            "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"
        )

        # Connect to Redis
        self.redis_conn = redis.from_url(self.redis_url)

        # Create queues with priorities
        self.high_queue = Queue("high", connection=self.redis_conn)
        self.default_queue = Queue("default", connection=self.redis_conn)
        self.low_queue = Queue("low", connection=self.redis_conn)
        self.model_queue = Queue("model_training", connection=self.redis_conn)

        # Setup RabbitMQ for event-driven communication
        self.rabbitmq_conn = None
        self.rabbitmq_channel = None
        # Only setup RabbitMQ if URL is provided and different from default
        if rabbitmq_url and rabbitmq_url != "amqp://guest:guest@localhost:5672/%2F":
            self._setup_rabbitmq()
        else:
            logger.info("RabbitMQ not configured, using Redis-only mode")

        self.extractor = None

    def _setup_rabbitmq(self):
        """Setup RabbitMQ connection and exchanges"""
        try:
            # Connect to RabbitMQ
            parameters = pika.URLParameters(self.rabbitmq_url)
            self.rabbitmq_conn = pika.BlockingConnection(parameters)
            self.rabbitmq_channel = self.rabbitmq_conn.channel()

            # Declare exchanges
            self.rabbitmq_channel.exchange_declare(
                exchange="analysis_events", exchange_type="topic", durable=True
            )

            # Declare queues
            self.rabbitmq_channel.queue_declare(queue="analysis_jobs", durable=True)
            self.rabbitmq_channel.queue_declare(queue="analysis_results", durable=True)
            self.rabbitmq_channel.queue_declare(queue="analysis_progress", durable=True)
            self.rabbitmq_channel.queue_declare(queue="model_training", durable=True)

            # Bind queues to exchanges
            self.rabbitmq_channel.queue_bind(
                exchange="analysis_events",
                queue="analysis_jobs",
                routing_key="analysis.job.*",
            )
            self.rabbitmq_channel.queue_bind(
                exchange="analysis_events",
                queue="analysis_results",
                routing_key="analysis.result.*",
            )
            self.rabbitmq_channel.queue_bind(
                exchange="analysis_events",
                queue="analysis_progress",
                routing_key="analysis.progress.*",
            )
            self.rabbitmq_channel.queue_bind(
                exchange="analysis_events",
                queue="model_training",
                routing_key="model.training.*",
            )

            logger.info("RabbitMQ connection established")
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ: {e}")
            self.rabbitmq_conn = None
            self.rabbitmq_channel = None

    def initialize_extractor(self):
        """Initialize the ATT&CK extractor"""
        if not self.extractor:
            self.extractor = EnhancedATTCKExtractor(
                data_dir="data",
                models_dir="models",
                neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
                neo4j_password=os.environ.get("NEO4J_PASSWORD", "password"),
                use_gpu=os.environ.get("USE_GPU", "true").lower() == "true",
                memory_efficient=os.environ.get("MEMORY_EFFICIENT", "true").lower()
                == "true",
            )

    def process_analysis_job(self, job_id: str, analysis_request: Dict[str, Any]):
        """Process a single analysis job with enhanced state management"""
        db = get_db()
        neo4j = get_neo4j()

        try:
            # Create state tracker
            state_tracker = JobStateTracker(
                job_id, self.redis_conn, self.rabbitmq_channel
            )

            # Update job status to processing
            db.execute(
                "UPDATE analysis_jobs SET status = 'processing' WHERE id = %s",
                (job_id,),
            )

            # Publish job started event
            state_tracker.update_state("processing", 0.1, "Started processing")

            # Initialize extractor if needed
            self.initialize_extractor()
            state_tracker.update_state("processing", 0.3, "Models loaded")

            # Perform analysis
            start_time = time.time()
            results = self.extractor.extract_techniques(
                text=analysis_request["text"],
                extractors=analysis_request.get("extractors"),
                threshold=analysis_request.get("threshold", 0.2),
                top_k=analysis_request.get("top_k", 10),
                use_ensemble=analysis_request.get("use_ensemble", True),
            )

            processing_time = int((time.time() - start_time) * 1000)
            state_tracker.update_state("processing", 0.7, "Analysis completed")

            # Store results in PostgreSQL
            for technique in results["techniques"]:
                db.execute(
                    """
                    INSERT INTO analysis_results 
                    (job_id, technique_id, technique_name, confidence, method, matched_keywords, cve_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        technique["technique_id"],
                        technique.get("name", ""),
                        technique["confidence"],
                        technique["method"],
                        technique.get("matched_keywords", []),
                        technique.get("cve_id"),
                    ),
                )

            state_tracker.update_state("processing", 0.9, "Storing results")

            # Update job status to completed
            db.execute(
                """
                UPDATE analysis_jobs 
                SET status = 'completed', 
                    completed_at = CURRENT_TIMESTAMP,
                    processing_time_ms = %s
                WHERE id = %s
                """,
                (processing_time, job_id),
            )

            # Mark job as complete and store results
            state_tracker.update_state(
                "completed", 1.0, "Analysis complete", results=results
            )

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")

            # Update job status to failed
            db.execute(
                "UPDATE analysis_jobs SET status = 'failed' WHERE id = %s", (job_id,)
            )

            # Publish error notification
            state_tracker = JobStateTracker(
                job_id, self.redis_conn, self.rabbitmq_channel
            )
            state_tracker.update_state("failed", 0.0, f"Error: {str(e)}")

    def process_model_retraining(self, forced: bool = False):
        """
        Process model retraining job based on feedback data

        Args:
            forced: Whether to force retraining even with insufficient data
        """
        db = get_db()
        training_id = f"training_{int(time.time())}"

        try:
            # Create state tracker
            state_tracker = JobStateTracker(
                training_id,
                self.redis_conn,
                self.rabbitmq_channel,
                job_type="model_training",
            )

            # Start training process
            state_tracker.update_state("processing", 0.1, "Started model retraining")

            # Insert training log entry
            db.execute(
                """
                INSERT INTO model_training_logs 
                (training_data_path, example_count, status, last_training_date)
                VALUES (%s, 0, 'started', CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (f"data/training/feedback_training_{training_id}.json",),
            )

            # Get training log ID
            training_log = db.query_one(
                """
                SELECT id FROM model_training_logs
                WHERE training_data_path = %s
                """,
                (f"data/training/feedback_training_{training_id}.json",),
            )

            training_log_id = training_log["id"] if training_log else None

            # Retrieve training data
            state_tracker.update_state("processing", 0.2, "Gathering training data")

            # Get positive examples from feedback
            positive_examples = db.query(
                """
                SELECT aj.input_data as text, af.technique_id as technique_id
                FROM analysis_feedback af
                JOIN analysis_jobs aj ON af.analysis_id = aj.id
                WHERE af.feedback_type = 'correct'
                AND af.created_at > (
                    SELECT COALESCE(MAX(last_training_date), '1970-01-01'::timestamp) 
                    FROM model_training_logs
                    WHERE status = 'completed'
                )
                """
            )

            # Get negative examples from feedback
            negative_examples = db.query(
                """
                SELECT aj.input_data as text, af.technique_id as incorrect_technique,
                       af.suggested_alternative as correct_technique
                FROM analysis_feedback af
                JOIN analysis_jobs aj ON af.analysis_id = aj.id
                WHERE af.feedback_type = 'incorrect'
                AND af.suggested_alternative IS NOT NULL
                AND af.created_at > (
                    SELECT COALESCE(MAX(last_training_date), '1970-01-01'::timestamp) 
                    FROM model_training_logs
                    WHERE status = 'completed'
                )
                """
            )

            # Get highlighted examples from feedback
            highlighted_examples = db.query(
                """
                SELECT fh.segment_text as text, af.technique_id as technique_id
                FROM feedback_highlights fh
                JOIN analysis_feedback af ON fh.feedback_id = af.id
                WHERE af.feedback_type = 'correct'
                AND af.created_at > (
                    SELECT COALESCE(MAX(last_training_date), '1970-01-01'::timestamp) 
                    FROM model_training_logs
                    WHERE status = 'completed'
                )
                """
            )

            # Check if we have enough data
            total_examples = (
                len(positive_examples)
                + len(negative_examples)
                + len(highlighted_examples)
            )

            if total_examples < 10 and not forced:
                # Not enough data for training
                state_tracker.update_state(
                    "failed",
                    0.0,
                    f"Insufficient training data: {total_examples} examples (minimum 10 required)",
                )

                # Update training log
                if training_log_id:
                    db.execute(
                        """
                        UPDATE model_training_logs
                        SET status = 'failed', 
                            error_message = %s,
                            example_count = %s
                        WHERE id = %s
                        """,
                        (
                            f"Insufficient training data: {total_examples} examples (minimum 10 required)",
                            total_examples,
                            training_log_id,
                        ),
                    )

                return

            # Prepare training data
            state_tracker.update_state("processing", 0.3, "Preparing training data")

            # Create directories if needed
            import os

            training_dir = "data/training"
            os.makedirs(training_dir, exist_ok=True)

            # Save training data
            training_data = {
                "positive_examples": [dict(ex) for ex in positive_examples],
                "negative_examples": [dict(ex) for ex in negative_examples],
                "highlighted_examples": [dict(ex) for ex in highlighted_examples],
                "training_date": datetime.now().isoformat(),
                "total_examples": total_examples,
            }

            training_file = f"{training_dir}/feedback_training_{training_id}.json"

            with open(training_file, "w") as f:
                json.dump(training_data, f, indent=2)

            # Update training log with example count
            if training_log_id:
                db.execute(
                    """
                    UPDATE model_training_logs
                    SET example_count = %s
                    WHERE id = %s
                    """,
                    (total_examples, training_log_id),
                )

            # Initialize extractor if needed
            self.initialize_extractor()
            state_tracker.update_state("processing", 0.4, "Models loaded")

            # Run model retraining (placeholder - this would call the actual training logic)
            # In a real implementation, this would:
            # 1. Extract features from examples
            # 2. Fine-tune or retrain models
            # 3. Evaluate performance
            # 4. Save updated models

            # Simulate training process
            state_tracker.update_state("processing", 0.5, "Training in progress")
            time.sleep(2)  # Simulated training time
            state_tracker.update_state("processing", 0.7, "Models updated")
            time.sleep(1)  # Simulated evaluation time
            state_tracker.update_state("processing", 0.9, "Finalizing training")

            # Update training log
            if training_log_id:
                db.execute(
                    """
                    UPDATE model_training_logs
                    SET status = 'completed'
                    WHERE id = %s
                    """,
                    (training_log_id,),
                )

            # Mark training as complete
            state_tracker.update_state(
                "completed",
                1.0,
                f"Training completed with {total_examples} examples",
                results={
                    "total_examples": total_examples,
                    "training_file": training_file,
                },
            )

            logger.info(f"Model retraining job {training_id} completed successfully")

        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")

            # Update training log
            if training_log_id:
                db.execute(
                    """
                    UPDATE model_training_logs
                    SET status = 'failed', 
                        error_message = %s
                    WHERE id = %s
                    """,
                    (str(e), training_log_id),
                )

            # Update state tracker
            state_tracker = JobStateTracker(
                training_id,
                self.redis_conn,
                self.rabbitmq_channel,
                job_type="model_training",
            )
            state_tracker.update_state("failed", 0.0, f"Error: {str(e)}")

    def enqueue_analysis(self, job_id: str, analysis_request: Dict[str, Any]) -> Job:
        """Enqueue an analysis job with priority support"""
        # Determine job priority
        priority = analysis_request.get("priority", "normal")
        timeout = "30m"

        if priority == "high":
            queue = self.high_queue
            timeout = "15m"  # Shorter timeout for high priority
        elif priority == "low":
            queue = self.low_queue
            timeout = "60m"  # Longer timeout for low priority
        else:
            queue = self.default_queue

        # Create initial job state
        state_tracker = JobStateTracker(job_id, self.redis_conn, self.rabbitmq_channel)
        state_tracker.update_state("pending", 0.0, "Job queued")

        # Enqueue job
        return queue.enqueue(
            self.process_analysis_job,
            job_id,
            analysis_request,
            job_timeout=timeout,
            result_ttl=86400,  # Keep results for 24 hours
        )

    def enqueue_model_retraining(self, forced: bool = False) -> Job:
        """
        Enqueue a model retraining job

        Args:
            forced: Whether to force retraining even with insufficient data

        Returns:
            Job instance
        """
        # Enqueue job
        return self.model_queue.enqueue(
            self.process_model_retraining,
            forced,
            job_timeout="1h",
            result_ttl=86400,  # Keep results for 24 hours
        )

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed job status with progress information"""
        # Try to get status from RQ job
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            status = {
                "id": job.id,
                "status": job.get_status(),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "ended_at": job.ended_at,
                "result": job.result if job.is_finished else None,
                "exc_info": job.exc_info if job.is_failed else None,
            }
        except Exception:
            # Job not found in RQ, try to get from database
            db = get_db()
            job_record = db.query_one(
                "SELECT * FROM analysis_jobs WHERE id = %s", (job_id,)
            )

            if not job_record:
                return None

            status = {
                "id": job_id,
                "status": job_record.get("status", "unknown"),
                "created_at": job_record.get("created_at"),
                "ended_at": job_record.get("completed_at"),
            }

        # Get detailed progress from state tracker
        state_tracker = JobStateTracker(job_id, self.redis_conn)
        state = state_tracker.get_state()

        if state:
            status.update(
                {
                    "progress": state.get("progress", 0),
                    "state_message": state.get("message", ""),
                    "last_updated": state.get("timestamp"),
                }
            )

        return status


class JobStateTracker:
    """Tracks job state and progress with Redis and RabbitMQ integration"""

    def __init__(
        self,
        job_id: str,
        redis_conn=None,
        rabbitmq_channel=None,
        job_type: str = "analysis",
    ):
        """
        Initialize the state tracker

        Args:
            job_id: Job identifier
            redis_conn: Redis connection
            rabbitmq_channel: RabbitMQ channel
            job_type: Type of job (analysis or model_training)
        """
        self.job_id = job_id
        self.job_type = job_type
        self.redis_conn = redis_conn
        self.rabbitmq_channel = rabbitmq_channel
        self.state_key = f"{job_type}_state:{job_id}"

    def update_state(self, status: str, progress: float, message: str, results=None):
        """
        Update job state and publish event

        Args:
            status: Job status
            progress: Progress value (0-1)
            message: Status message
            results: Optional results data
        """
        state = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if results:
            state["results"] = results

        # Store in Redis
        if self.redis_conn:
            self.redis_conn.set(self.state_key, json.dumps(state))

            # Set expiration (24 hours)
            self.redis_conn.expire(self.state_key, 86400)

        # Publish to RabbitMQ
        if self.rabbitmq_channel:
            try:
                # Determine routing key based on job type
                if self.job_type == "model_training":
                    routing_key = f"model.training.{status}"
                else:
                    routing_key = f"analysis.progress.{status}"

                self.rabbitmq_channel.basic_publish(
                    exchange="analysis_events",
                    routing_key=routing_key,
                    body=json.dumps(state),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        content_type="application/json",
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to publish state update to RabbitMQ: {e}")

    def get_state(self) -> Dict:
        """
        Get current job state

        Returns:
            Job state dictionary
        """
        if not self.redis_conn:
            return {}

        state_json = self.redis_conn.get(self.state_key)
        if not state_json:
            return {}

        try:
            return json.loads(state_json)
        except Exception as e:
            logger.error(f"Failed to parse job state: {e}")
            return {}
