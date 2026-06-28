"""
modules/chatbot.py
==================
LLM-powered HR Assistant chatbot using Google Gemini API (new google-genai SDK).

Responsibilities
----------------
1. Build a rich system prompt from pipeline results (rankings, metrics, raw CVs).
2. Maintain conversational memory within a Streamlit session.
3. Stream responses from the Gemini API for a responsive chat experience.

Design Principles
-----------------
- Context is built once per pipeline run and cached in session state.
- Raw CV Markdown is included for deep analysis (Gemini handles large contexts well).
- The system prompt instructs the LLM to act as an expert HR screening assistant.

External Dependencies
---------------------
- google-genai   : Google Gemini API client (new SDK)
- python-dotenv  : Load API key from .env

Configuration (.env)
--------------------
    GEMINI_API_KEY = "your_gemini_api_key_here"
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Gemini Client Setup ──────────────────────────────────────────────────────

_gemini_client = None


def _get_gemini_client():
    """
    Lazily initialize and cache the Gemini client.

    Uses the new google-genai SDK which replaces the deprecated
    google-generativeai package.

    Returns:
        A configured ``google.genai.Client`` instance.

    Raises:
        ImportError: If the google-genai package is not installed.
        ValueError: If the GEMINI_API_KEY environment variable is not set.
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai not installed. "
            "Run: pip install google-genai"
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set in .env file. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    _gemini_client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized (google-genai SDK).")
    return _gemini_client


# ── System Prompt Builder ─────────────────────────────────────────────────────

def build_system_prompt(
    results_df: pd.DataFrame,
    candidates: list,
    job_description: str,
    filtered: Optional[list] = None,
) -> str:
    """
    Build a comprehensive system prompt for the HR Assistant chatbot.

    Combines multiple data sources into a structured prompt:
    1. Role definition and behavioral guidelines for the LLM.
    2. The Job Description being recruited for.
    3. A formatted metrics table (Rank, Name, Fit Score, CGPA, Skills).
    4. Full Markdown text of each candidate's resume for deep analysis.
    5. List of filtered-out candidates with rejection reasons.

    Args:
        results_df: DataFrame from the ranking pipeline with candidate scores.
        candidates: List of CandidateRecord objects with resume_markdown populated.
        job_description: The full job description text.
        filtered: Optional list of CandidateRecord objects that failed filters.

    Returns:
        A single string containing the full system prompt.
    """
    # ── Part 1: Role definition ───────────────────────────────────────────────
    prompt_parts = [
        "# ROLE: Expert HR Screening Assistant for CVision",
        "",
        "You are an AI HR Assistant embedded in the CVision resume screening platform.",
        "Your job is to help the recruiter analyze, compare, and make decisions about",
        "candidates based on their resumes, fit scores, and qualifications.",
        "",
        "## Guidelines:",
        "- Always reference specific data from the candidates when answering.",
        "- Be concise but thorough. Use bullet points and tables when helpful.",
        "- When comparing candidates, cite their exact scores, skills, and experience.",
        "- If asked to recommend candidates, explain your reasoning clearly.",
        "- You can identify strengths, weaknesses, red flags, and skill gaps.",
        "- When uncertain, say so rather than making things up.",
        "- Format your responses in clean Markdown.",
        "",
    ]

    # ── Part 2: Job Description ───────────────────────────────────────────────
    prompt_parts.extend([
        "---",
        "# JOB DESCRIPTION (What we are hiring for)",
        "",
        job_description.strip(),
        "",
    ])

    # ── Part 3: Rankings & Metrics Table ──────────────────────────────────────
    prompt_parts.extend([
        "---",
        "# CANDIDATE RANKINGS & METRICS",
        "",
        "The following candidates have been ranked by semantic similarity to the",
        "job description using a BERT embedding model. Higher Fit Score = better match.",
        "",
    ])

    # Build a clean table from the results DataFrame
    display_cols = [c for c in [
        "Rank", "Candidate Name", "Fit Score (%)", "CGPA", "Degree", "Top Skills"
    ] if c in results_df.columns]

    if display_cols:
        # Markdown table header
        header = "| " + " | ".join(display_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
        prompt_parts.append(header)
        prompt_parts.append(separator)

        for _, row in results_df.iterrows():
            cells = []
            for col in display_cols:
                val = row.get(col, "N/A")
                if col == "Fit Score (%)":
                    cells.append(f"{val:.1f}%" if isinstance(val, (int, float)) else str(val))
                else:
                    cells.append(str(val))
            prompt_parts.append("| " + " | ".join(cells) + " |")

        prompt_parts.append("")

    # ── Part 4: Per-candidate contact info + email ────────────────────────────
    prompt_parts.extend([
        "---",
        "# CANDIDATE CONTACT INFORMATION",
        "",
    ])

    # Build a lookup for quick access
    record_by_name = {c.name: c for c in candidates}

    for _, row in results_df.iterrows():
        cand_name = row.get("Candidate Name", "")
        record = record_by_name.get(cand_name)
        if record:
            prompt_parts.append(f"**{record.name}**")
            prompt_parts.append(f"- Email: {record.email or 'N/A'}")
            prompt_parts.append(f"- Phone: {record.phone or 'N/A'}")
            prompt_parts.append(f"- CGPA: {record.cgpa if record.cgpa != -1.0 else 'N/A'}")
            prompt_parts.append("")

    # ── Part 5: Full Resume Markdown (for deep analysis) ──────────────────────
    prompt_parts.extend([
        "---",
        "# FULL RESUME TEXTS (for detailed analysis)",
        "",
        "Below are the complete extracted texts from each candidate's resume.",
        "Use these for detailed questions about experience, projects, education, etc.",
        "",
    ])

    for _, row in results_df.iterrows():
        cand_name = row.get("Candidate Name", "")
        record = record_by_name.get(cand_name)
        if record and record.resume_markdown:
            # Truncate extremely long CVs to ~4000 chars each to manage context
            md_text = record.resume_markdown[:4000]
            if len(record.resume_markdown) > 4000:
                md_text += "\n\n[... truncated for brevity ...]"
            prompt_parts.extend([
                f"## Resume: {record.name}",
                "",
                md_text,
                "",
            ])

    # ── Part 6: Filtered-out candidates ───────────────────────────────────────
    if filtered:
        prompt_parts.extend([
            "---",
            "# FILTERED OUT CANDIDATES (failed hard filters)",
            "",
        ])
        for c in filtered:
            prompt_parts.append(
                f"- **{c.name}** ({c.email}): {c.filter_reason}"
            )
        prompt_parts.append("")

    return "\n".join(prompt_parts)


# ── Chat Function ─────────────────────────────────────────────────────────────

def chat_with_assistant(
    user_message: str,
    chat_history: list[dict],
    system_prompt: str,
) -> str:
    """
    Send a user message to the Gemini LLM and return the assistant's response.

    Uses the new google-genai SDK. Maintains conversation context by passing
    the full chat history to the model. The system prompt (containing all
    candidate data) is prepended as the first message in the conversation.

    Args:
        user_message: The user's question or instruction.
        chat_history: List of {"role": "user"|"model", "parts": [text]} dicts
                      representing the conversation so far.
        system_prompt: The full system prompt built by ``build_system_prompt()``.

    Returns:
        The assistant's text response.

    Raises:
        Exception: On API errors (quota exceeded, network issues, etc.).
    """
    client = _get_gemini_client()

    # Build the full conversation for Gemini
    # System prompt goes as the first user message with context framing
    contents = [
        {"role": "user", "parts": [{"text": system_prompt + "\n\n---\nAcknowledge that you have received and understood all the candidate data above. From now on, answer questions about these candidates."}]},
        {"role": "model", "parts": [{"text": "I have received and thoroughly analyzed all the candidate data, including their rankings, fit scores, CGPA, skills, contact information, and full resume texts. I'm ready to help you with detailed insights, comparisons, and recommendations. What would you like to know?"}]},
    ]

    # Append existing conversation history (convert to new format)
    for msg in chat_history:
        parts_data = msg.get("parts", [])
        contents.append({
            "role": msg["role"],
            "parts": [{"text": p} if isinstance(p, str) else p for p in parts_data],
        })

    # Append the new user message
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config={
                "temperature": 0.4,
                "top_p": 0.95,
                "max_output_tokens": 4096,
            },
        )
        assistant_reply = response.text
        logger.info("Gemini response received (%d chars).", len(assistant_reply))
        return assistant_reply
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise
