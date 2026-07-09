"""
modules/email_dispatch.py
=========================
Automated email dispatch for candidate acceptance and rejection notifications.

Responsibilities
----------------
1. Send polite rejection emails to candidates who failed hard filters.
2. Send invitation emails to candidates selected for the next round.
3. Send rejection emails to candidates not selected after final review.

Design Principles
-----------------
- Uses Gmail SMTP with App Passwords (secure, no OAuth flow needed).
- All emails are templated with candidate name substitution.
- Provides a dry-run mode (preview emails without sending) for testing.

Configuration (.env)
--------------------
    SMTP_EMAIL        = "your_gmail@gmail.com"
    SMTP_APP_PASSWORD = "your_16_char_app_password"
    SMTP_HOST         = "smtp.gmail.com"        # default
    SMTP_PORT         = 465                      # default (SSL)
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Email Templates ──────────────────────────────────────────────────────────

REJECTION_FILTER_TEMPLATE = """
Dear {name},

Thank you for taking the time to apply for the {position} position at our organization.

After careful review of your application, we regret to inform you that we are unable to move forward with your candidacy at this time, as your application did not meet our minimum screening requirements.

Reason: {reason}

We encourage you to continue developing your skills and qualifications, and we welcome you to apply for future openings that may be a better fit.

We wish you the best in your career endeavors.

Warm regards,
{company_name} Recruitment Team
""".strip()

REJECTION_FINAL_TEMPLATE = """
Dear {name},

Thank you for your interest in the {position} position and for submitting your application.

After a thorough review of all candidates, we have decided to proceed with other applicants whose qualifications more closely align with our current requirements.

This was a competitive process, and your application was carefully considered. We encourage you to apply for future positions with us.

We wish you all the best in your career.

Warm regards,
{company_name} Recruitment Team
""".strip()

ACCEPTANCE_TEMPLATE = """
Dear {name},

Congratulations! We are pleased to inform you that your application for the {position} position has been shortlisted, and we would like to invite you to the next round of our selection process.

Your qualifications and experience stood out among the applicants, and we are excited to learn more about you.

We will be reaching out shortly with details regarding the next steps, including scheduling and any preparation materials.

If you have any questions in the meantime, please don't hesitate to reach out.

We look forward to speaking with you!

Warm regards,
{company_name} Recruitment Team
""".strip()


# ── SMTP Configuration ───────────────────────────────────────────────────────

def _get_smtp_config() -> dict:
    """
    Read SMTP configuration from environment variables.

    Returns:
        Dict with keys: email, password, host, port.

    Raises:
        ValueError: If required SMTP_EMAIL or SMTP_APP_PASSWORD are not set.
    """
    email = os.getenv("SMTP_EMAIL", "").strip()
    password = os.getenv("SMTP_APP_PASSWORD", "").strip()
    host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.getenv("SMTP_PORT", "465"))

    if not email or not password:
        raise ValueError(
            "SMTP_EMAIL and SMTP_APP_PASSWORD must be set in .env file.\n"
            "For Gmail, generate an App Password at: "
            "https://myaccount.google.com/apppasswords"
        )

    return {"email": email, "password": password, "host": host, "port": port}


# ── Email Sending ─────────────────────────────────────────────────────────────

def send_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_config: Optional[dict] = None,
    dry_run: bool = False,
) -> bool:
    """
    Send a single email via Gmail SMTP (SSL).

    Args:
        to_email: Recipient email address.
        subject: Email subject line.
        body: Plain-text email body.
        smtp_config: SMTP settings dict (auto-loaded from .env if None).
        dry_run: If True, log the email content but don't actually send.

    Returns:
        True if the email was sent (or dry-run logged) successfully.
        False if sending failed.
    """
    if dry_run:
        logger.info(
            "[DRY RUN] Would send email:\n  To: %s\n  Subject: %s\n  Body: %s...",
            to_email, subject, body[:200],
        )
        return True

    if not smtp_config:
        try:
            smtp_config = _get_smtp_config()
        except ValueError as e:
            logger.error("SMTP config missing: %s", e)
            return False

    msg = MIMEMultipart()
    msg["From"] = smtp_config["email"]
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_config["host"], smtp_config["port"], context=context) as server:
            server.login(smtp_config["email"], smtp_config["password"])
            server.send_message(msg)
        logger.info("Native SMTP email sent to: %s — Subject: %s", to_email, subject)
        return True
    except Exception as exc:
        logger.error("Failed to send native SMTP email to %s: %s", to_email, exc)
        return False


# ── Batch Email Functions ─────────────────────────────────────────────────────

def send_filter_rejection_emails(
    filtered_candidates: list,
    position: str = "the open position",
    company_name: str = "Our Organization",
    dry_run: bool = True,
) -> dict:
    """
    Send rejection emails to all candidates who failed the hard knockout filters.

    Args:
        filtered_candidates: List of CandidateRecord objects with passed_filter=False.
        position: Job title for the email template.
        company_name: Company name for the email template.
        dry_run: If True, preview emails without sending.

    Returns:
        Dict with keys 'sent', 'failed', 'skipped' counting results.
    """
    results = {"sent": 0, "failed": 0, "skipped": 0}

    smtp_config = None
    if not dry_run:
        try:
            smtp_config = _get_smtp_config()
        except ValueError as exc:
            logger.error("Cannot send emails: %s", exc)
            return results

    for candidate in filtered_candidates:
        if not candidate.email:
            logger.warning("No email for filtered candidate: %s — skipped.", candidate.name)
            results["skipped"] += 1
            continue

        body = REJECTION_FILTER_TEMPLATE.format(
            name=candidate.name,
            position=position,
            reason=candidate.filter_reason or "Did not meet minimum requirements.",
            company_name=company_name,
        )

        success = send_email(
            to_email=candidate.email,
            subject=f"Application Update — {position}",
            body=body,
            smtp_config=smtp_config,
            dry_run=dry_run,
        )

        if success:
            results["sent"] += 1
        else:
            results["failed"] += 1

    logger.info(
        "Filter rejection emails: %d sent, %d failed, %d skipped.",
        results["sent"], results["failed"], results["skipped"],
    )
    return results


def send_final_decision_emails(
    selected: list,
    rejected: list,
    position: str = "the open position",
    company_name: str = "Our Organization",
    dry_run: bool = True,
) -> dict:
    """
    Send acceptance emails to selected candidates and rejection emails to the rest.

    Args:
        selected: List of CandidateRecord objects chosen for the next round.
        rejected: List of CandidateRecord objects not selected.
        position: Job title for the email template.
        company_name: Company name for the email template.
        dry_run: If True, preview emails without sending.

    Returns:
        Dict with keys 'accepted_sent', 'rejected_sent', 'failed', 'skipped'.
    """
    results = {"accepted_sent": 0, "rejected_sent": 0, "failed": 0, "skipped": 0}

    smtp_config = None
    if not dry_run:
        try:
            smtp_config = _get_smtp_config()
        except ValueError as exc:
            logger.error("Cannot send emails: %s", exc)
            return results

    # ── Acceptance emails ─────────────────────────────────────────────────────
    for candidate in selected:
        if not candidate.email:
            results["skipped"] += 1
            continue

        body = ACCEPTANCE_TEMPLATE.format(
            name=candidate.name,
            position=position,
            company_name=company_name,
        )

        success = send_email(
            to_email=candidate.email,
            subject=f"🎉 Congratulations! Next Round Invitation — {position}",
            body=body,
            smtp_config=smtp_config,
            dry_run=dry_run,
        )

        if success:
            results["accepted_sent"] += 1
        else:
            results["failed"] += 1

    # ── Rejection emails ──────────────────────────────────────────────────────
    for candidate in rejected:
        if not candidate.email:
            results["skipped"] += 1
            continue

        body = REJECTION_FINAL_TEMPLATE.format(
            name=candidate.name,
            position=position,
            company_name=company_name,
        )

        success = send_email(
            to_email=candidate.email,
            subject=f"Application Update — {position}",
            body=body,
            smtp_config=smtp_config,
            dry_run=dry_run,
        )

        if success:
            results["rejected_sent"] += 1
        else:
            results["failed"] += 1

    logger.info(
        "Final decision emails: %d accepted, %d rejected, %d failed, %d skipped.",
        results["accepted_sent"], results["rejected_sent"],
        results["failed"], results["skipped"],
    )
    return results
