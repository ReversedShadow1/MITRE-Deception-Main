# src/queue/worker.py
import logging
import os
import sys

from redis import Redis
from rq import Connection, Queue, Worker

from src.queue.manager import AnalysisQueueManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("worker")


def main():
    """Worker process entry point"""
    # Get Redis URL from environment
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    # Initialize Redis connection
    redis_conn = Redis.from_url(redis_url)

    # Initialize queue manager
    queue_manager = AnalysisQueueManager(redis_url)

    # Get worker type from environment
    worker_type = os.environ.get("WORKER_TYPE", "default")

    # Determine queue names based on worker type
    queue_names = ["default"]

    if worker_type == "all":
        queue_names = ["high", "default", "low", "model_training"]
    elif worker_type == "high_priority":
        queue_names = ["high"]
    elif worker_type == "model_training":
        queue_names = ["model_training"]
    elif worker_type == "analysis":
        queue_names = ["high", "default", "low"]

    # Start worker
    with Connection(redis_conn):
        queues = [Queue(name) for name in queue_names]
        worker = Worker(queues)

        logger.info(f"Starting {worker_type} worker with PID {os.getpid()}")
        logger.info(f"Watching queues: {', '.join(q.name for q in queues)}")

        worker.work()


if __name__ == "__main__":
    main()


def start_worker_scaling():
    """Start worker scaling service"""
    from src.queue.auto_scaling import WorkerScaler

    # Initialize and start the scaler
    scaler = WorkerScaler(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379"),
        min_workers=2,
        max_workers=10,
    )
    scaler.start()

    return scaler
