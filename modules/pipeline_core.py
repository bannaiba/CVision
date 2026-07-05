import logging
import tempfile
from pathlib import Path

import pandas as pd
from typing import Callable, Optional

from modules.ingestion import (
    CandidateRecord,
    fetch_candidates_from_sheet,
    apply_knockout_filters,
    download_all_resumes,
)
from modules.parser import extract_markdown_from_pdf
from modules.embedding import rank_resumes_semantic, compute_summary_stats

logger = logging.getLogger(__name__)

def run_headless_sheet_pipeline(
    jd: str,
    sheet_id: str,
    credentials_path: str,
    min_cgpa: float,
    min_years_exp: float,
    model,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, dict, list[CandidateRecord], list[CandidateRecord]]:
    """
    Headless version of the Google Sheet pipeline.
    Suitable for execution in background threads (e.g. apscheduler).
    Raises exceptions on failure instead of using Streamlit UI calls.
    """
    if progress_callback: progress_callback("📊 Fetching candidates from Google Sheet...")
    logger.info("Fetching candidates from Google Sheet...")
    all_candidates = fetch_candidates_from_sheet(sheet_id, credentials_path)
    
    if not all_candidates:
        raise ValueError("No candidate submissions found in the Google Sheet.")

    if progress_callback: progress_callback("🚫 Applying knockout filters...")
    logger.info("Applying knockout filters...")
    apply_knockout_filters(all_candidates, min_cgpa=min_cgpa, min_years_exp=min_years_exp)

    passed    = [c for c in all_candidates if c.passed_filter]
    filtered  = [c for c in all_candidates if not c.passed_filter]

    if not passed:
        logger.warning("All candidates were filtered out.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="cvision_"))
    if progress_callback: progress_callback("📥 Downloading resumes from Google Drive...")
    logger.info("Downloading resumes from Google Drive...")
    
    if passed:
        download_all_resumes(
            candidates=passed,
            dest_dir=tmp_dir,
            credentials_path=credentials_path,
            delay_seconds=0.3,
        )

    downloaded  = [c for c in passed if c.local_resume_path]
    no_download = [c for c in passed if not c.local_resume_path]

    if no_download:
        logger.warning(f"Could not download {len(no_download)} resume(s).")

    if passed and not downloaded:
        raise RuntimeError("No resumes could be downloaded. Check Drive permissions.")

    if progress_callback: progress_callback("📄 Extracting text from resumes...")
    logger.info("Extracting text from resumes...")
    filenames:        list[str] = []
    resume_markdowns: list[str] = []
    candidate_meta:   list[dict] = []

    for candidate in downloaded:
        markdown = extract_markdown_from_pdf(candidate.local_resume_path)
        candidate.resume_markdown = markdown   # Store on record for future use

        if markdown.strip():
            filenames.append(candidate.local_resume_path.name)
            resume_markdowns.append(markdown)
            candidate_meta.append({
                "name":      candidate.name,
                "email":     candidate.email,
                "cgpa":      candidate.cgpa,
                "years_exp": candidate.years_exp,
                "degree":    candidate.degree,
            })
        else:
            logger.warning(f"Empty extraction for: {candidate.name} — skipped from ranking.")

    if downloaded and not filenames:
        raise RuntimeError("Text extraction failed for all downloaded resumes.")

    if progress_callback: progress_callback("🧠 Running BERT semantic ranking...")
    logger.info("Running BERT semantic ranking...")
    if filenames:
        results_df = rank_resumes_semantic(
            job_description=jd,
            filenames=filenames,
            resume_markdowns=resume_markdowns,
            model=model,
            candidate_metadata=candidate_meta,
        )
        stats = compute_summary_stats(results_df)
    else:
        results_df = pd.DataFrame()
        stats = compute_summary_stats(results_df)

    return results_df, stats, downloaded, filtered
