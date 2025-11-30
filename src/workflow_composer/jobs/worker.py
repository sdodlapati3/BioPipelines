"""
Celery Worker Entrypoint.

Usage:
    celery -A src.workflow_composer.jobs.worker worker --loglevel=info
    
For specific queues:
    celery -A src.workflow_composer.jobs.worker worker -Q workflows,search --loglevel=info
    
With beat scheduler:
    celery -A src.workflow_composer.jobs.worker beat --loglevel=info
"""

import os
import logging
from src.workflow_composer.jobs.celery_app import celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Import tasks to ensure they're registered
from src.workflow_composer.jobs import tasks  # noqa: F401


def start_worker(
    queues: list = None,
    concurrency: int = None,
    loglevel: str = "INFO",
):
    """
    Start a Celery worker programmatically.
    
    Args:
        queues: List of queues to consume from
        concurrency: Number of concurrent workers
        loglevel: Logging level
    """
    argv = ["worker"]
    
    if queues:
        argv.extend(["-Q", ",".join(queues)])
    
    if concurrency:
        argv.extend(["-c", str(concurrency)])
    
    argv.extend(["--loglevel", loglevel])
    
    celery_app.worker_main(argv=argv)


def start_beat(loglevel: str = "INFO"):
    """
    Start the Celery beat scheduler programmatically.
    
    Args:
        loglevel: Logging level
    """
    celery_app.Beat(
        app=celery_app,
        loglevel=loglevel,
    ).run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Celery Worker for BioPipelines")
    parser.add_argument(
        "--queues", "-Q",
        help="Comma-separated list of queues to consume",
        default="default,workflows,search,validation,execution",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        help="Number of concurrent workers",
        default=4,
    )
    parser.add_argument(
        "--loglevel", "-l",
        help="Logging level",
        default="INFO",
    )
    parser.add_argument(
        "--beat", "-B",
        action="store_true",
        help="Also run the beat scheduler",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Celery worker with queues: {args.queues}")
    logger.info(f"Concurrency: {args.concurrency}")
    
    queues = args.queues.split(",") if args.queues else None
    start_worker(
        queues=queues,
        concurrency=args.concurrency,
        loglevel=args.loglevel,
    )
