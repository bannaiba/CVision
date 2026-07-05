import logging
import json
import pickle
import os
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


def _load_scheduler_config() -> dict | None:
    """
    Load scheduler configuration from multiple sources (priority order):
    1. SCHEDULER_CONFIG env var (JSON string — persists on Render).
    2. scheduler_config.json file on disk (works locally).
    Returns None if neither source is available.
    """
    # 1. Try environment variable first (cloud-friendly)
    raw = os.getenv("SCHEDULER_CONFIG", "").strip()
    if raw:
        try:
            config = json.loads(raw)
            logger.info("Loaded scheduler config from SCHEDULER_CONFIG env var.")
            return config
        except json.JSONDecodeError:
            logger.error("SCHEDULER_CONFIG env var contains invalid JSON.")

    # 2. Fall back to local file
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded scheduler config from {CONFIG_FILE}.")
            return config
        except Exception as e:
            logger.error(f"Failed to read {CONFIG_FILE}: {e}")

    return None


def save_scheduler_config(config: dict) -> str:
    """
    Save scheduler configuration to both file (local) and return
    the JSON string so the UI can display instructions for Render.
    """
    # Always save to local file
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    # Also set it as an env var for the current process
    # (so the scheduler can pick it up immediately)
    config_json = json.dumps(config)
    os.environ["SCHEDULER_CONFIG"] = config_json

    return config_json


def _scheduler_job():
    """
    The job that the apscheduler runs. It reads the config, runs the headless pipeline,
    auto-sends emails if toggled, and saves the cache for Streamlit to pick up.
    """
    logger.info("Scheduler Job Triggered: Starting automated pipeline run.")

    config = _load_scheduler_config()
    if config is None:
        logger.error("Cannot run scheduled job: no configuration found.")
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

    On startup, if a SCHEDULER_CONFIG env var exists with a schedule_time,
    the scheduler will auto-configure itself so it survives Render restarts.
    """
    scheduler = BackgroundScheduler(timezone="Asia/Dhaka")
    scheduler.start()

    # Auto-restore scheduled job from env var on startup
    config = _load_scheduler_config()
    if config and config.get("schedule_time") and config.get("schedule_enabled", False):
        time_str = config["schedule_time"]
        try:
            hour, minute = map(int, time_str.split(":"))
            trigger = CronTrigger(hour=hour, minute=minute, timezone="Asia/Dhaka")
            scheduler.add_job(
                _scheduler_job, trigger=trigger,
                id="daily_pipeline_run", replace_existing=True
            )
            logger.info(f"Auto-restored scheduled job at {hour:02d}:{minute:02d} UTC from saved config.")
        except Exception as e:
            logger.error(f"Failed to auto-restore schedule: {e}")

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
        trigger = CronTrigger(hour=hour, minute=minute, timezone="Asia/Dhaka")
        scheduler.add_job(_scheduler_job, trigger=trigger, id=job_id, replace_existing=True)
        logger.info(f"Scheduled pipeline to run daily at {hour:02d}:{minute:02d} UTC")
    else:
        logger.info("Scheduled pipeline disabled.")
