import logging
import json
import pickle
import os
import threading
from pathlib import Path
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from modules.pipeline_core import run_headless_sheet_pipeline
from modules.embedding import load_embedding_model
from modules.email_dispatch import send_filter_rejection_emails

logger = logging.getLogger(__name__)

CACHE_FILE = ".pipeline_cache.pkl"
CONFIG_FILE = "scheduler_config.json"

def _scheduler_job():
    """
    The job that the apscheduler runs. It reads the config, runs the headless pipeline,
    auto-sends emails if toggled, and saves the cache for Streamlit to pick up.
    """
    logger.info("Scheduler Job Triggered: Starting automated pipeline run.")
    
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Cannot run scheduled job: {CONFIG_FILE} not found.")
        return

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {CONFIG_FILE}: {e}")
        return

    # Extract config
    jd = config.get("job_description", "")
    sheet_id = config.get("sheet_id", "")
    min_cgpa = config.get("min_cgpa", 0.0)
    min_years_exp = config.get("min_years_exp", 0.0)
    credentials_path = config.get("credentials_path", "credentials.json")
    model_name = config.get("model_name", "all-MiniLM-L6-v2")
    auto_email = config.get("auto_email", False)

    if not jd or not sheet_id:
        logger.error("Missing required config (job_description or sheet_id). Aborting scheduled run.")
        return

    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = load_embedding_model(model_name)
        
        results_df, stats, candidates, filtered = run_headless_sheet_pipeline(
            jd=jd,
            sheet_id=sheet_id,
            credentials_path=credentials_path,
            min_cgpa=min_cgpa,
            min_years_exp=min_years_exp,
            model=model
        )

        logger.info("Pipeline successful. Caching results...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "results_df": results_df,
                "stats": stats,
                "candidates": candidates,
                "filtered": filtered,
                "job_description": jd,
                "last_run": datetime.now().isoformat()
            }, f)

        if auto_email and filtered:
            logger.info(f"Auto-sending filter rejections for {len(filtered)} candidates...")
            send_filter_rejection_emails(
                filtered_candidates=filtered,
                position="the open position",
                company_name="Our Organization",
                dry_run=False,
            )

        logger.info("Scheduled job completed successfully.")

    except Exception as e:
        logger.exception(f"Scheduled job failed: {e}")


def get_scheduler() -> BackgroundScheduler:
    """
    Returns the singleton background scheduler.
    Intended to be cached by Streamlit so it persists across reruns.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.start()
    return scheduler

def update_scheduler_job(scheduler: BackgroundScheduler, time_str: str, enabled: bool):
    """
    Updates the daily scheduled job. 
    `time_str` is expected to be in HH:MM format (24-hour).
    """
    job_id = "daily_pipeline_run"
    
    # Remove existing job if any
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    if enabled and time_str:
        hour, minute = map(int, time_str.split(":"))
        trigger = CronTrigger(hour=hour, minute=minute)
        scheduler.add_job(_scheduler_job, trigger=trigger, id=job_id, replace_existing=True)
        logger.info(f"Scheduled pipeline to run daily at {hour:02d}:{minute:02d} UTC")
    else:
        logger.info("Scheduled pipeline disabled.")
