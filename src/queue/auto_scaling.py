import logging
import os
import threading
import time
from typing import Dict, List

import redis
from rq import Queue, Worker
from rq.job import Job

logger = logging.getLogger(__name__)


class WorkerScaler:
    """
    Dynamically scales the number of worker processes based on queue load
    """

    def __init__(
        self,
        redis_url: str = None,
        min_workers: int = 2,
        max_workers: int = 10,
        target_job_ratio: float = 2.0,  # Jobs per worker
        scale_up_threshold: float = 3.0,  # Scale up when jobs/worker exceeds this
        scale_down_threshold: float = 1.0,  # Scale down when jobs/worker below this
        check_interval: int = 30,  # Check every 30 seconds
    ):
        """Initialize the worker scaler"""
        self.redis_url = redis_url or os.environ.get(
            "REDIS_URL", "redis://localhost:6379"
        )
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_job_ratio = target_job_ratio
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.check_interval = check_interval

        # Connect to Redis
        self.redis_conn = redis.from_url(self.redis_url)

        # Initialize queues
        self.queues = {
            "high": Queue("high", connection=self.redis_conn),
            "default": Queue("default", connection=self.redis_conn),
            "low": Queue("low", connection=self.redis_conn),
            "model_training": Queue("model_training", connection=self.redis_conn),
        }

        # Worker registry
        self.workers = {}
        self.workers_lock = threading.Lock()

        # Start monitor thread
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)

        logger.info(f"Worker scaler initialized (min={min_workers}, max={max_workers})")

    def start(self):
        """Start the monitoring thread and ensure minimum workers"""
        logger.info("Starting worker scaler")

        # Ensure minimum workers
        self._ensure_min_workers()

        # Start monitor thread
        self.monitor_thread.start()

        logger.info(f"Worker scaler started with {len(self.workers)} workers")

    def stop(self):
        """Stop the monitoring thread and workers"""
        logger.info("Stopping worker scaler")

        # Stop monitor thread
        self.stop_event.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # Stop all workers
        with self.workers_lock:
            for worker_id, worker_info in list(self.workers.items()):
                self._stop_worker(worker_id)

        logger.info("Worker scaler stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                self._check_and_scale()
            except Exception as e:
                logger.error(f"Error in worker scaler monitor: {e}")

            # Wait for next check
            for _ in range(self.check_interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

    def _check_and_scale(self):
        """Check queue status and scale workers as needed"""
        # Get queue lengths
        queue_lengths = {name: queue.count for name, queue in self.queues.items()}
        total_jobs = sum(queue_lengths.values())

        # Get current worker count
        with self.workers_lock:
            current_workers = len(self.workers)

        # Calculate jobs per worker ratio
        if current_workers > 0:
            jobs_per_worker = total_jobs / current_workers
        else:
            jobs_per_worker = float("inf")  # Infinite ratio if no workers

        logger.info(
            f"Queue status: {queue_lengths}, Workers: {current_workers}, Ratio: {jobs_per_worker:.1f}"
        )

        # Determine if we need to scale
        if (
            jobs_per_worker > self.scale_up_threshold
            and current_workers < self.max_workers
        ):
            # Scale up
            target_workers = min(
                self.max_workers,
                current_workers + 1,
                max(self.min_workers, int(total_jobs / self.target_job_ratio)),
            )
            if target_workers > current_workers:
                self._scale_to(target_workers)

        elif (
            jobs_per_worker < self.scale_down_threshold
            and current_workers > self.min_workers
        ):
            # Scale down
            target_workers = max(
                self.min_workers,
                min(current_workers - 1, int(total_jobs / self.target_job_ratio) + 1),
            )
            if target_workers < current_workers:
                self._scale_to(target_workers)

    def _ensure_min_workers(self):
        """Ensure minimum number of workers are running"""
        with self.workers_lock:
            current_workers = len(self.workers)
            if current_workers < self.min_workers:
                for _ in range(self.min_workers - current_workers):
                    self._start_worker()

    def _scale_to(self, target_count: int):
        """Scale to the target number of workers"""
        with self.workers_lock:
            current_count = len(self.workers)

            if target_count > current_count:
                # Scale up
                logger.info(
                    f"Scaling up from {current_count} to {target_count} workers"
                )
                for _ in range(target_count - current_count):
                    self._start_worker()

            elif target_count < current_count:
                # Scale down
                logger.info(
                    f"Scaling down from {current_count} to {target_count} workers"
                )
                # Sort workers by idle time (descending) to remove most idle workers first
                workers_by_idle = sorted(
                    self.workers.items(),
                    key=lambda x: x[1].get("idle_since", 0),
                    reverse=True,
                )

                # Remove excess workers
                for worker_id, _ in workers_by_idle[: current_count - target_count]:
                    self._stop_worker(worker_id)

    def _start_worker(self):
        """Start a new worker process"""
        import subprocess
        import uuid

        worker_id = str(uuid.uuid4())

        # Determine environment variables
        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["REDIS_URL"] = self.redis_url

        # Start worker process
        try:
            process = subprocess.Popen(
                ["python", "-m", "src.queue.worker"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Register worker
            with self.workers_lock:
                self.workers[worker_id] = {
                    "process": process,
                    "started_at": time.time(),
                    "idle_since": None,
                }

            logger.info(f"Started worker {worker_id} (pid={process.pid})")

            # Start monitor thread for this worker
            threading.Thread(
                target=self._monitor_worker, args=(worker_id, process), daemon=True
            ).start()

            return worker_id

        except Exception as e:
            logger.error(f"Error starting worker: {e}")
            return None

    def _stop_worker(self, worker_id: str):
        """Stop a worker process"""
        with self.workers_lock:
            if worker_id not in self.workers:
                return

            worker_info = self.workers[worker_id]
            process = worker_info.get("process")

            if process:
                try:
                    # Try graceful shutdown first
                    process.terminate()

                    # Wait for process to exit
                    for _ in range(10):  # Wait up to 10 seconds
                        if process.poll() is not None:
                            break
                        time.sleep(1)

                    # Force kill if still running
                    if process.poll() is None:
                        process.kill()

                    logger.info(f"Stopped worker {worker_id}")

                except Exception as e:
                    logger.error(f"Error stopping worker {worker_id}: {e}")

            # Remove from registry
            del self.workers[worker_id]

    def _monitor_worker(self, worker_id: str, process):
        """Monitor a worker process and handle its output and termination"""
        # Read output
        for line in iter(process.stdout.readline, b""):
            try:
                line_str = line.decode("utf-8").strip()
                if line_str:
                    logger.debug(f"Worker {worker_id}: {line_str}")

                    # Update idle status based on output
                    if "Waiting for jobs" in line_str:
                        with self.workers_lock:
                            if worker_id in self.workers:
                                self.workers[worker_id]["idle_since"] = time.time()
                    elif "Processing job" in line_str:
                        with self.workers_lock:
                            if worker_id in self.workers:
                                self.workers[worker_id]["idle_since"] = None
            except Exception as e:
                logger.error(f"Error processing worker output: {e}")

        # Process ended
        return_code = process.wait()
        logger.info(f"Worker {worker_id} exited with code {return_code}")

        # Remove from registry if it's still there
        with self.workers_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]

        # Start a new worker if we're below minimum
        self._ensure_min_workers()

    def get_status(self) -> Dict:
        """Get current worker status"""
        with self.workers_lock:
            return {
                "workers": {
                    "current": len(self.workers),
                    "min": self.min_workers,
                    "max": self.max_workers,
                },
                "queues": {name: queue.count for name, queue in self.queues.items()},
                "total_jobs": sum(queue.count for queue in self.queues.values()),
                "workers_detail": [
                    {
                        "id": worker_id,
                        "pid": info["process"].pid if info["process"] else None,
                        "uptime": time.time() - info["started_at"]
                        if info["started_at"]
                        else 0,
                        "idle": info["idle_since"] is not None,
                        "idle_time": time.time() - info["idle_since"]
                        if info["idle_since"]
                        else 0,
                    }
                    for worker_id, info in self.workers.items()
                ],
            }
