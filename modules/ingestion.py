"""
modules/ingestion.py
====================
Data ingestion layer for the CVision resume screening pipeline.

Responsibilities
----------------
1. Fetch candidate application rows from a Google Sheet linked to a Google Form.
2. Apply hard-coded knockout filters (CGPA, years of experience, degree level).
3. Download resume PDF files from Google Drive for candidates who pass the filters.
4. Provide a realistic mock dataset for offline/demo use when no credentials exist.

Design Principles
-----------------
- Stateless pure functions wherever possible (easy to unit-test in isolation).
- CandidateRecord is the single data contract shared across all pipeline stages.
- Every external call (Sheets API, Drive API, HTTP) has a graceful fallback.
- Candidates who fail filters are kept in the list (passed_filter=False) so the
  UI can render a "Rejected at screening" panel — important for auditability.

External Dependencies
---------------------
- gspread                   : Google Sheets API client
- google-auth               : Service Account authentication
- google-api-python-client  : Google Drive file download
- python-dotenv             : Load secrets from .env
- requests                  : HTTP fallback for public Drive URLs

Configuration (.env)
--------------------
    GOOGLE_SHEET_ID          = "1abc..."     # From Sheet URL /d/THIS_ID/edit
    GOOGLE_CREDENTIALS_PATH  = "credentials.json"   # Service Account JSON key

See GOOGLE_FORM_SPEC.md for the exact Google Form structure and
Service Account setup walkthrough.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()  # Load .env file if present (no-op if absent)

logger = logging.getLogger(__name__)

# ── Credentials Resolution ────────────────────────────────────────────────────

_CACHED_CREDENTIALS_PATH: Optional[str] = None

def resolve_credentials_path(credentials_path: str = "credentials.json") -> str:
    """
    Resolve the Google Service Account credentials to a file path.

    Priority:
    1. If ``credentials_path`` points to an existing file, use it directly.
    2. If the env var ``GOOGLE_CREDENTIALS_JSON`` contains the raw JSON string,
       write it to a temporary file and return that path. This is the recommended
       approach for cloud platforms like Render where you cannot deploy files.
    3. Raise FileNotFoundError if neither source is available.
    """
    global _CACHED_CREDENTIALS_PATH

    # 1. Check if the file already exists on disk (local dev)
    if Path(credentials_path).exists():
        return credentials_path

    # 2. Check if we already wrote a temp file in a previous call
    if _CACHED_CREDENTIALS_PATH and Path(_CACHED_CREDENTIALS_PATH).exists():
        return _CACHED_CREDENTIALS_PATH

    # 3. Try to load from environment variable
    raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    if raw_json:
        try:
            # Validate that it's proper JSON
            json.loads(raw_json)
            # Write to a temp file that persists for the process lifetime
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="gcp_creds_", delete=False
            )
            tmp.write(raw_json)
            tmp.close()
            _CACHED_CREDENTIALS_PATH = tmp.name
            logger.info("Loaded Google credentials from GOOGLE_CREDENTIALS_JSON env var.")
            return _CACHED_CREDENTIALS_PATH
        except json.JSONDecodeError:
            logger.error("GOOGLE_CREDENTIALS_JSON env var contains invalid JSON.")

    raise FileNotFoundError(
        f"Credentials file not found: {credentials_path}\n"
        "Either place credentials.json in the project folder, or set the "
        "GOOGLE_CREDENTIALS_JSON environment variable with the raw JSON content.\n"
        "See GOOGLE_FORM_SPEC.md for Service Account setup instructions."
    )

# ── Knockout Filter Defaults ──────────────────────────────────────────────────

DEFAULT_MIN_CGPA: float = 3.0
"""Minimum acceptable CGPA on a 4.0 scale."""

DEFAULT_MIN_YEARS_EXP: float = 3.0
"""Minimum years of professional experience."""

DEFAULT_ALLOWED_DEGREES: tuple[str, ...] = (
    "bachelor",
    "master",
    "phd",
    "doctorate",
    "bs",
    "ms",
    "be",
    "bsc",
    "msc",
    "beng",
    "meng",
    "b.s",
    "m.s",
    "b.e",
    "b.sc",
    "m.sc",
)
"""Lowercase substrings that identify an acceptable degree tier (Bachelor's or higher)."""

# ── Google Form → Sheet Column Mapping ───────────────────────────────────────
# Keys are internal logical names; values are the EXACT question text from the
# Google Form (which becomes the Google Sheet column header).
# If you rename a form question, update the value here to match.
COLUMN_MAP: dict[str, str] = {
    "timestamp":  "Timestamp",
    "name":       "Full Name",
    "email":      "Email Address",
    "phone":      "Phone Number",
    "cgpa":       "CGPA/GPA",
    "degree":     "Highest Degree Earned",
    "major":      "Major/Field of Study",
    "years_exp":  "Years of Professional Experience",
    "job_title":  "Current or Last Job Title",
    "linkedin":   "LinkedIn Profile URL",
    "position":   "Position Applying For",
    "resume_url": "Upload Resume/CV",
    "cover_note": "Brief Cover Note",
}


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    """
    Immutable snapshot of a single candidate's application.

    This dataclass acts as the **central data contract** that flows through every
    stage of the pipeline: ingestion → parser → embedding → (future) communication.
    All downstream modules receive a list of CandidateRecord objects and read from
    them; they never write back except via the designated mutable fields below.

    Attributes
    ----------
    name : str
        Candidate's full name as entered in the Google Form.
    email : str
        Email address (will be used for automated communication in Phase 2).
    phone : str
        Phone number; may be empty if the optional form field was skipped.
    cgpa : float
        GPA on a 4.0 scale. Set to -1.0 if the form value was not parseable.
    degree : str
        Highest degree earned, lowercased from form dropdown value.
    major : str
        Field of study.
    years_exp : float
        Years of professional experience. -1.0 if not parseable.
    job_title : str
        Current or last job title; may be empty.
    linkedin_url : str
        LinkedIn profile URL; may be empty.
    position_applied : str
        Role the candidate is applying for.
    resume_drive_url : str
        Raw Google Drive URL of the uploaded resume PDF from the form response.
    cover_note : str
        Optional free-text cover note from the form.
    local_resume_path : Optional[Path]
        Set by download_all_resumes() once the PDF is on disk. None until then.
    resume_markdown : str
        Set by the parser module after PDF text extraction. Empty until then.
    passed_filter : bool
        True if the candidate passed all knockout filters. Default True.
    filter_reason : str
        Human-readable rejection reason if passed_filter is False.
    fit_score : float
        BERT cosine similarity score [0–100]. Set by the embedding module.
    top_skills : list[str]
        Technical skills extracted from the resume. Set by the embedding module.
    """

    # ── Form data ──────────────────────────────────────────────────────────────
    name: str
    email: str
    phone: str = ""
    cgpa: float = -1.0
    degree: str = ""
    major: str = ""
    years_exp: float = -1.0
    job_title: str = ""
    linkedin_url: str = ""
    position_applied: str = ""
    resume_drive_url: str = ""
    cover_note: str = ""

    # ── Mutable pipeline state (set by downstream modules) ─────────────────────
    local_resume_path: Optional[Path] = field(default=None, repr=False)
    resume_markdown: str = field(default="", repr=False)
    passed_filter: bool = True
    filter_reason: str = ""
    fit_score: float = 0.0
    top_skills: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a plain dict of display-relevant fields for DataFrame construction."""
        return {
            "Name": self.name,
            "Email": self.email,
            "CGPA": self.cgpa if self.cgpa != -1.0 else "N/A",
            "Degree": self.degree.title() if self.degree else "N/A",
            "Years Exp.": self.years_exp if self.years_exp != -1.0 else "N/A",
            "Position": self.position_applied,
            "Passed Filter": self.passed_filter,
            "Filter Reason": self.filter_reason,
        }


# ── Google Sheets: Authentication ─────────────────────────────────────────────

def _get_gspread_client(credentials_path: str, readonly: bool = True):
    """
    Authenticate with Google and return a gspread Client.

    Uses a Service Account JSON key (recommended for server-side apps because
    it does not require user interaction and credentials never expire).

    Args:
        credentials_path: Filesystem path to the Service Account JSON key file
                          downloaded from Google Cloud Console.

    Returns:
        An authenticated ``gspread.Client`` instance.

    Raises:
        ImportError: If gspread or google-auth packages are not installed.
        FileNotFoundError: If credentials_path does not point to an existing file.
        google.auth.exceptions.GoogleAuthError: If authentication fails (e.g.,
            wrong scopes, revoked credentials).
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise ImportError(
            "Google API packages not installed. "
            "Run: pip install gspread google-auth google-api-python-client"
        ) from exc

    resolved_path = resolve_credentials_path(credentials_path)

    # Request only the minimum necessary scopes (principle of least privilege)
    if readonly:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
    else:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
    creds = Credentials.from_service_account_file(resolved_path, scopes=scopes)
    client = gspread.authorize(creds)
    logger.debug("gspread authenticated successfully with service account.")
    return client

def export_results_to_sheet(
    sheet_id: str,
    credentials_path: str,
    results_df,
    candidates: list,
    selected_names: list,
    email_status: dict,
    tab_prefix: str = "CVision Database"
):
    """
    Export the analysis results to a Google Sheet.
    Creates the tab if it does not exist.
    """
    import datetime
    from zoneinfo import ZoneInfo
    client = _get_gspread_client(credentials_path, readonly=False)
    sheet = client.open_by_key(sheet_id)
    
    bd_tz = ZoneInfo("Asia/Dhaka")
    now_bd = datetime.datetime.now(bd_tz)
    timestamp = now_bd.strftime("%Y-%m-%d %H.%M.%S")
    tab_name = f"{tab_prefix} - {timestamp}"
    
    try:
        worksheet = sheet.add_worksheet(title=tab_name, rows="100", cols="20")
    except Exception as e:
        logger.error(f"Failed to create new worksheet {tab_name}: {e}")
        worksheet = sheet.worksheet(tab_prefix) # Fallback to default
    
    # Define headers
    headers = ["Status", "Candidate Name", "Phone", "CGPA", "Rank", "Fit Score (%)", "Top Skills", "Degree", "Email Sent", "Saved At"]
    
    phone_map = {c.name: (c.phone or "N/A") for c in candidates}
    
    # We can use a more readable timestamp format for the rows
    row_timestamp = now_bd.strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare rows
    rows_to_insert = [headers]
    for _, row in results_df.iterrows():
        cand_name = row.get("Candidate Name", row.get("Filename", "Unknown"))
        status = "Accepted" if cand_name in selected_names else "Rejected"
        phone = phone_map.get(cand_name, "N/A")
        cgpa = row.get("CGPA", "")
        rank = row.get("Rank", "")
        score = row.get("Fit Score (%)", "")
        skills = row.get("Top Skills", "")
        degree = row.get("Degree", "")
        email_sent = email_status.get(cand_name, "No")
        
        rows_to_insert.append([status, cand_name, phone, cgpa, rank, score, skills, degree, email_sent, row_timestamp])
        
    worksheet.update(values=rows_to_insert, range_name="A1")

# ── Google Sheets: Data Fetching ──────────────────────────────────────────────

def fetch_candidates_from_sheet(
    sheet_id: str,
    credentials_path: str,
) -> list[CandidateRecord]:
    """
    Fetch all form submission rows from a Google Sheet and parse them.

    Expects the sheet's first row to be a header produced by Google Forms,
    with column names exactly matching the values in ``COLUMN_MAP``.

    Args:
        sheet_id: The Google Sheet ID (the long string in the URL between
                  ``/d/`` and ``/edit``).
        credentials_path: Path to Service Account JSON key file.

    Returns:
        List of ``CandidateRecord`` instances — one per non-empty form row.
        Rows that cannot be parsed are skipped with a warning log.

    Raises:
        gspread.exceptions.APIError: On Google API errors (quota, auth, etc.).
        ValueError: If the sheet appears empty (no rows beyond header).
    """
    logger.info("Connecting to Google Sheet ID: %s", sheet_id)

    try:
        client = _get_gspread_client(credentials_path)
        # .sheet1 accesses the first tab of the spreadsheet
        sheet = client.open_by_key(sheet_id).sheet1
        # get_all_records() returns List[dict] with header row as keys
        rows = sheet.get_all_records(
            empty2zero=False,
            head=1,
            default_blank="",
        )
    except Exception as exc:
        logger.error("Failed to read Google Sheet: %s", exc)
        raise

    if not rows:
        logger.warning("Google Sheet appears empty — no candidate submissions found.")
        return []

    candidates: list[CandidateRecord] = []
    for row_index, row in enumerate(rows, start=2):  # Row 1 is header
        try:
            candidate = _row_to_candidate(row)
            # Skip completely empty rows (e.g., blank rows at bottom of sheet)
            if not candidate.name and not candidate.email:
                continue
            candidates.append(candidate)
        except Exception as exc:
            logger.warning("Skipping row %d — parse error: %s", row_index, exc)

    logger.info("Fetched %d candidate record(s) from Google Sheet.", len(candidates))
    return candidates


def _row_to_candidate(row: dict) -> CandidateRecord:
    """
    Convert a single Google Sheet row dict into a CandidateRecord.

    Column headers in ``row`` are expected to match the values in ``COLUMN_MAP``.
    Missing or malformed values are handled gracefully with sensible defaults.

    Args:
        row: Dict mapping column headers to cell values for one submission row.

    Returns:
        A populated ``CandidateRecord`` instance.
    """

    def _get(key: str) -> str:
        """Look up a cell by logical key name; return stripped string or ''."""
        col_header = COLUMN_MAP.get(key, key)
        return str(row.get(col_header, "") or "").strip()

    def _parse_float(raw: str, default: float = -1.0) -> float:
        """
        Parse a string to float with multiple fallback strategies.

        Handles:
        - Plain numbers: "3.5" → 3.5
        - Range strings: "3-5 years" → 3.0 (take lower bound)
        - Numbers with trailing words: "4 years" → 4.0
        """
        raw = raw.strip()
        if not raw:
            return default
        # Extract first numeric token (int or float)
        match = re.search(r"(\d+\.?\d*)", raw)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return default

    return CandidateRecord(
        name=_get("name"),
        email=_get("email"),
        phone=_get("phone"),
        cgpa=_parse_float(_get("cgpa")),
        degree=_get("degree").lower(),
        major=_get("major"),
        years_exp=_parse_float(_get("years_exp")),
        job_title=_get("job_title"),
        linkedin_url=_get("linkedin"),
        position_applied=_get("position"),
        resume_drive_url=_get("resume_url"),
        cover_note=_get("cover_note"),
    )


# ── Knockout Filters ──────────────────────────────────────────────────────────

def apply_knockout_filters(
    candidates: list[CandidateRecord],
    min_cgpa: float = DEFAULT_MIN_CGPA,
    min_years_exp: float = DEFAULT_MIN_YEARS_EXP,
    allowed_degrees: tuple[str, ...] = DEFAULT_ALLOWED_DEGREES,
) -> list[CandidateRecord]:
    """
    Apply hard knockout filters and mark each candidate as pass or fail.

    Filters are applied in order; the **first failing filter** sets the reason and
    no further filters are evaluated for that candidate. Candidates are **never
    removed** from the list — they remain with ``passed_filter=False`` so the
    dashboard can display a separate "Screened Out" table with rejection reasons.

    Current filter rules (in evaluation order):
    1. CGPA ≥ min_cgpa (default 3.0 / 4.0)
    2. Years of experience ≥ min_years_exp (default 3)
    3. Degree is Bachelor's-level or higher

    Args:
        candidates: List of ``CandidateRecord`` instances to evaluate.
        min_cgpa: Minimum acceptable CGPA. Inclusive threshold.
        min_years_exp: Minimum years of professional experience. Inclusive.
        allowed_degrees: Tuple of lowercase substrings identifying qualifying degrees.

    Returns:
        The same list (mutated in-place) for chaining convenience.
        Each record's ``passed_filter`` and ``filter_reason`` fields are updated.
    """
    for candidate in candidates:
        # Reset filter state in case this function is called multiple times
        candidate.passed_filter = True
        candidate.filter_reason = ""

        # ── Filter 1: CGPA ────────────────────────────────────────────────────
        if candidate.cgpa == -1.0:
            # CGPA was not parseable — log but do NOT reject. Give benefit of doubt
            # since some applicants may enter formats we haven't anticipated.
            logger.debug("CGPA not parseable for '%s' — CGPA filter skipped.", candidate.name)
        elif candidate.cgpa < min_cgpa:
            candidate.passed_filter = False
            candidate.filter_reason = (
                f"CGPA {candidate.cgpa:.2f} is below the minimum requirement of {min_cgpa:.2f}"
            )
            continue  # Skip remaining filters once one fails

        # ── Filter 2: Years of Experience ────────────────────────────────────
        if candidate.years_exp == -1.0:
            logger.debug("Years of experience not parseable for '%s' — filter skipped.", candidate.name)
        elif candidate.years_exp < min_years_exp:
            candidate.passed_filter = False
            candidate.filter_reason = (
                f"Only {candidate.years_exp:.1f} year(s) of experience — "
                f"minimum required is {min_years_exp:.0f}"
            )
            continue

        # ── Filter 3: Degree Level ────────────────────────────────────────────
        if candidate.degree:
            degree_qualifies = any(
                allowed_deg in candidate.degree for allowed_deg in allowed_degrees
            )
            if not degree_qualifies:
                candidate.passed_filter = False
                candidate.filter_reason = (
                    f"Degree '{candidate.degree.title()}' does not meet the "
                    f"minimum qualification of a Bachelor's degree or higher"
                )
                continue

    passed_count = sum(1 for c in candidates if c.passed_filter)
    logger.info(
        "Knockout filter complete: %d/%d candidate(s) passed.",
        passed_count,
        len(candidates),
    )
    return candidates


# ── Google Drive: PDF Download ─────────────────────────────────────────────────

def _extract_drive_file_id(url: str) -> Optional[str]:
    """
    Extract the Google Drive file ID from any supported Drive URL format.

    Handles all common URL patterns produced by Google Forms file uploads:
    - ``https://drive.google.com/file/d/{ID}/view?usp=drivesdk``
    - ``https://drive.google.com/open?id={ID}``
    - ``https://docs.google.com/...?id={ID}``
    - Raw 28–33 character alphanumeric file ID (passed through directly)

    Args:
        url: A string that may be a Google Drive URL or a raw file ID.

    Returns:
        The file ID string if extraction succeeds, or ``None`` otherwise.
    """
    if not url or not url.strip():
        return None

    url = url.strip()

    # Pattern 1: /file/d/{ID}/  (most common for uploaded files)
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    # Pattern 2: ?id={ID} or &id={ID}  (older Drive links)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    # Pattern 3: Raw file ID — 28–40 alphanumeric chars
    if re.match(r"^[a-zA-Z0-9_-]{20,50}$", url):
        return url

    logger.debug("Could not extract Drive file ID from: %s", url[:80])
    return None


def download_resume_from_drive(
    drive_url: str,
    dest_dir: Path,
    candidate_name: str,
    credentials_path: Optional[str] = None,
    timeout: int = 30,
) -> Optional[Path]:
    """
    Download a resume PDF from Google Drive to a local directory.

    Download strategy (tried in order):
    1. **Drive API (authenticated)** — works for any file the service account
       can access. Required when the file is not publicly shared.
    2. **Direct export URL (public)** — falls back to a ``drive.google.com/uc``
       URL. Only works if the file is shared as "Anyone with the link".

    Args:
        drive_url: Google Drive URL or file ID from the Google Sheet response cell.
        dest_dir: Local directory where the downloaded PDF will be saved.
        candidate_name: Used to generate a safe filename (e.g., ``John_Doe.pdf``).
        credentials_path: Path to Service Account JSON. If None or file missing,
                          skips to the public URL fallback.
        timeout: HTTP request timeout in seconds for the public URL method.

    Returns:
        ``Path`` to the saved PDF file, or ``None`` if all download attempts failed.
    """
    file_id = _extract_drive_file_id(drive_url)
    if not file_id:
        logger.warning(
            "Cannot download resume for '%s': unrecognised Drive URL: %s",
            candidate_name, drive_url[:80],
        )
        return None

    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build a filesystem-safe filename from the candidate's name
    safe_name = re.sub(r"[^\w\s-]", "", candidate_name).strip().replace(" ", "_")
    safe_name = re.sub(r"_+", "_", safe_name) or "candidate"
    dest_path = dest_dir / f"{safe_name}.pdf"

    # ── Method 1: Authenticated Drive API ────────────────────────────────────
    resolved_path = None
    if credentials_path:
        try:
            resolved_path = resolve_credentials_path(credentials_path)
        except FileNotFoundError:
            pass

    if resolved_path:
        try:
            logger.info(
                "Attempting authenticated Drive download for '%s' (file_id=%s)...",
                candidate_name, file_id,
            )
            return _download_via_drive_api(file_id, dest_path, resolved_path)
        except Exception as exc:
            logger.warning(
                "Drive API download failed for '%s': %s — falling back to public URL.",
                candidate_name, exc,
            )

    # ── Method 2: Public export URL ──────────────────────────────────────────
    # Note: this only works if the file sharing is set to "Anyone with the link"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        logger.info(
            "Attempting public URL download for '%s'...", candidate_name
        )
        session = requests.Session()
        resp = session.get(download_url, stream=True, timeout=timeout)
        resp.raise_for_status()

        # Google sometimes returns an HTML virus-scan warning page for large files.
        # Detect this by checking Content-Type and retry with the confirmation token.
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            token_match = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
            if token_match:
                confirmed_url = f"{download_url}&confirm={token_match.group(1)}"
                resp = session.get(confirmed_url, stream=True, timeout=timeout)
                resp.raise_for_status()
            else:
                logger.warning(
                    "Drive returned HTML (not PDF) for '%s'. "
                    "Check that the file is publicly shared or credentials are provided.",
                    candidate_name,
                )
                return None

        # Stream the file to disk
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)

        # Sanity check: valid PDFs are at least a few KB
        if dest_path.stat().st_size < 1_024:
            logger.warning(
                "Downloaded file for '%s' is suspiciously small (%d bytes) — "
                "likely an error page, not a PDF.",
                candidate_name, dest_path.stat().st_size,
            )
            dest_path.unlink(missing_ok=True)
            return None

        logger.info("Downloaded resume: '%s' → %s", candidate_name, dest_path.name)
        return dest_path

    except requests.exceptions.RequestException as exc:
        logger.error("HTTP download failed for '%s': %s", candidate_name, exc)
        return None


def _download_via_drive_api(
    file_id: str,
    dest_path: Path,
    credentials_path: str,
) -> Path:
    """
    Download a file from Google Drive using the authenticated Drive v3 REST API.

    This method works regardless of the file's sharing settings as long as the
    Service Account has been granted at least Viewer access to the file or its
    parent folder.

    Args:
        file_id: Google Drive file ID (28–33 alphanumeric characters).
        dest_path: Full local path where the file content will be written.
        credentials_path: Path to the Service Account JSON key file.

    Returns:
        The ``dest_path`` after a successful write.

    Raises:
        googleapiclient.errors.HttpError: On API permission or not-found errors.
        Any exception from google-auth on credential loading failures.
    """
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    resolved_path = resolve_credentials_path(credentials_path)
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_file(resolved_path, scopes=scopes)

    # cache_discovery=False avoids a deprecation warning in newer library versions
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    dest_path.write_bytes(buffer.read())
    logger.debug("Drive API download complete → %s", dest_path)
    return dest_path


def download_all_resumes(
    candidates: list[CandidateRecord],
    dest_dir: Path,
    credentials_path: Optional[str] = None,
    delay_seconds: float = 0.5,
) -> list[CandidateRecord]:
    """
    Batch-download resume PDFs for all candidates who passed knockout filters.

    Updates each qualifying ``CandidateRecord.local_resume_path`` in-place.
    Candidates without a Drive URL or who failed filters are silently skipped.

    Args:
        candidates: Full list of ``CandidateRecord`` instances.
        dest_dir: Directory to save downloaded PDFs.
        credentials_path: Optional Service Account JSON path.
        delay_seconds: Polite delay between Drive API calls to avoid rate limits.
                       Set to 0 to disable (useful in tests).

    Returns:
        The input ``candidates`` list (mutated in-place) for chaining.
    """
    eligible = [
        c for c in candidates
        if c.passed_filter and c.resume_drive_url.strip()
    ]

    if not eligible:
        logger.warning(
            "No eligible candidates with Drive URLs found for download. "
            "If using Upload mode, this is expected."
        )
        return candidates

    logger.info("Starting resume download for %d eligible candidate(s).", len(eligible))

    for idx, candidate in enumerate(eligible, start=1):
        logger.info(
            "Downloading [%d/%d]: %s", idx, len(eligible), candidate.name
        )
        path = download_resume_from_drive(
            drive_url=candidate.resume_drive_url,
            dest_dir=dest_dir,
            candidate_name=candidate.name,
            credentials_path=credentials_path,
        )
        candidate.local_resume_path = path

        if path is None:
            logger.warning("⚠ Resume unavailable for: %s", candidate.name)

        # Polite delay between Drive API requests
        if delay_seconds > 0 and idx < len(eligible):
            time.sleep(delay_seconds)

    downloaded = sum(1 for c in eligible if c.local_resume_path)
    logger.info(
        "Resume download complete: %d/%d succeeded.", downloaded, len(eligible)
    )
    return candidates


# ── Mock Dataset (Demo / Offline Mode) ────────────────────────────────────────

def load_mock_candidates() -> list[CandidateRecord]:
    """
    Return a curated list of realistic mock candidates for demo and offline use.

    Intentionally includes:
    - Candidates who will pass all filters (majority)
    - One who fails the CGPA filter
    - One who fails the experience filter
    This allows the filtering UI and "Screened Out" panel to be demonstrated
    without any real submissions or API calls.

    Returns:
        List of 6 ``CandidateRecord`` instances with no Drive URLs
        (``local_resume_path`` must be populated manually if testing with real PDFs).
    """
    return [
        CandidateRecord(
            name="Aisha Rahman",
            email="aisha.rahman@email.com",
            phone="+92-300-1234567",
            cgpa=3.8,
            degree="master's degree",
            major="Computer Science",
            years_exp=5.0,
            job_title="Senior ML Engineer",
            linkedin_url="https://linkedin.com/in/aisha-rahman",
            position_applied="AI Engineer",
            resume_drive_url="",
        ),
        CandidateRecord(
            name="Bilal Ahmed",
            email="bilal.ahmed@email.com",
            phone="+92-321-9876543",
            cgpa=3.2,
            degree="bachelor's degree",
            major="Software Engineering",
            years_exp=3.5,
            job_title="Data Scientist",
            linkedin_url="https://linkedin.com/in/bilal-ahmed",
            position_applied="AI Engineer",
            resume_drive_url="",
        ),
        CandidateRecord(
            name="Sara Khan",
            email="sara.khan@email.com",
            phone="+92-333-5554444",
            cgpa=2.7,          # ← Will FAIL CGPA filter (< 3.0)
            degree="bachelor's degree",
            major="Electrical Engineering",
            years_exp=4.0,
            job_title="Software Developer",
            linkedin_url="",
            position_applied="AI Engineer",
            resume_drive_url="",
        ),
        CandidateRecord(
            name="Omar Sheikh",
            email="omar.sheikh@email.com",
            phone="+92-311-2223333",
            cgpa=3.5,
            degree="phd",
            major="Artificial Intelligence",
            years_exp=2.0,     # ← Will FAIL experience filter (< 3)
            job_title="Research Associate",
            linkedin_url="https://linkedin.com/in/omar-sheikh",
            position_applied="AI Engineer",
            resume_drive_url="",
        ),
        CandidateRecord(
            name="Fatima Malik",
            email="fatima.malik@email.com",
            phone="+92-345-6789012",
            cgpa=3.6,
            degree="master's degree",
            major="Data Science",
            years_exp=6.0,
            job_title="Lead Data Scientist",
            linkedin_url="https://linkedin.com/in/fatima-malik",
            position_applied="AI Engineer",
            resume_drive_url="",
        ),
        CandidateRecord(
            name="Hassan Raza",
            email="hassan.raza@email.com",
            phone="+92-322-1112222",
            cgpa=3.1,
            degree="bachelor's degree",
            major="Computer Engineering",
            years_exp=4.5,
            job_title="ML Engineer",
            linkedin_url="https://linkedin.com/in/hassan-raza",
            position_applied="AI Engineer",
            resume_drive_url="",
        ),
    ]

def export_results_to_sheet(sheet_id: str, credentials_path: str, results_df: pd.DataFrame, candidates: list, selected_names: list, email_status: dict, tab_prefix: str = "CVision Database") -> None:
    """
    Exports the analyzed results to a new tab in the Google Sheet.
    Includes a timestamp of when it was saved.
    """
    from datetime import datetime, timezone, timedelta
    import gspread
    
    client = _get_gspread_client(credentials_path, readonly=False)
    # Ignore the input sheet_id and hardcode to the Export Database
    sheet = client.open_by_key("1NJurIfA-q9J5ifr_cc7KQJH8L9xQjbwhQdH2_nUp7gQ")
    
    # Bangladesh Time is UTC+6
    bst_tz = timezone(timedelta(hours=6))
    timestamp_str = datetime.now(bst_tz).strftime("%Y-%m-%d %H:%M:%S")
    worksheet_title = f"{tab_prefix} - {timestamp_str}"
    
    export_df = results_df.copy()
    export_df["Export Timestamp"] = timestamp_str
    
    # Fill NA values so gspread doesn't crash on NaN
    export_df = export_df.fillna("")
    
    data = [export_df.columns.values.tolist()] + export_df.values.tolist()
    
    try:
        worksheet = sheet.add_worksheet(title=worksheet_title, rows=max(100, len(data) + 10), cols=max(20, len(export_df.columns) + 5))
    except gspread.exceptions.APIError as e:
        # If worksheet with this exact timestamp exists (unlikely), append a random suffix
        import random
        worksheet_title += f"_{random.randint(100,999)}"
        worksheet = sheet.add_worksheet(title=worksheet_title, rows=max(100, len(data) + 10), cols=max(20, len(export_df.columns) + 5))

    worksheet.update(values=data, range_name="A1")

def export_chat_to_sheet(credentials_path: str, chat_messages: list, tab_name: str = "CVision Chat Logs") -> None:
    """
    Appends the current chat transcript to a dedicated tab in the Google Sheet.
    """
    from datetime import datetime, timezone, timedelta
    import gspread
    
    if not chat_messages:
        raise ValueError("No chat history found. The transcript is empty.")
        
    client = _get_gspread_client(credentials_path, readonly=False)
    sheet = client.open_by_key("1NJurIfA-q9J5ifr_cc7KQJH8L9xQjbwhQdH2_nUp7gQ")
    
    try:
        worksheet = sheet.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=tab_name, rows=1000, cols=5)
        worksheet.append_row(["Timestamp", "Chat Transcript"])
        
    bst_tz = timezone(timedelta(hours=6))
    timestamp_str = datetime.now(bst_tz).strftime("%Y-%m-%d %H:%M:%S")
    
    transcript_lines = []
    for msg in chat_messages:
        role = msg.get("role", "unknown").upper()
        parts = msg.get("parts", [""])
        content = parts[0] if parts else ""
        if role == "USER":
            transcript_lines.append(f"👤 [USER]:\n{content}")
        else:
            transcript_lines.append(f"🤖 [AGENT]:\n{content}")
            
    full_transcript = "\n\n---\n\n".join(transcript_lines)
    
    worksheet.append_row([timestamp_str, full_transcript])
