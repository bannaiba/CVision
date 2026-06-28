"""
modules/parser.py
=================
Layout-aware PDF-to-Markdown text extraction for resume files.

Why pdfplumber over the original pypdf?
----------------------------------------
The original app.py used pypdf, which extracts text in a simple stream and
loses all spatial information. This works for single-column PDFs but fails
completely on the multi-column, visually dense layouts common in resumes —
it reads columns left-to-right across the page, mixing content from different
sections into a single garbled string.

pdfplumber gives us character-level bounding boxes (x0, y0, x1, y1) and
font size per character cluster. This enables us to:

  1. Detect multi-column layout by clustering word X-coordinates.
  2. Reconstruct correct reading order (finish left column before right).
  3. Identify section headings via font size, ALL-CAPS, and keyword matching.
  4. Preserve bullet point structure as Markdown ``- item`` syntax.
  5. Produce clean, token-efficient Markdown that the sentence-transformers
     model can embed with full semantic context.

The fallback chain is: pdfplumber → pypdf → empty string (with warnings).
This ensures the pipeline never crashes on a problematic PDF.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Type alias for the three forms a PDF source can take
PDFSource = Union[str, Path, io.BytesIO]

# ── Section Heading Keywords ───────────────────────────────────────────────────
# Used as a strong signal when deciding if a short line is a section heading.
# Any line containing one of these substrings (case-insensitive) gets a +2 score.
_HEADING_KEYWORDS: frozenset[str] = frozenset({
    "experience", "education", "skills", "projects", "certifications",
    "summary", "objective", "awards", "publications", "languages",
    "interests", "references", "achievements", "work history",
    "technical skills", "professional experience", "academic",
    "profile", "contact", "about", "training", "courses", "career",
    "qualifications", "employment", "background",
})


# ── Public API ────────────────────────────────────────────────────────────────

def extract_markdown_from_pdf(
    pdf_source: PDFSource,
    *,
    fallback_to_pypdf: bool = True,
) -> str:
    """
    Extract text from a PDF resume and return it as structured Markdown.

    This is the primary entry point for the parser module. It attempts
    pdfplumber extraction first (layout-aware, column-detecting) and falls
    back to pypdf if that fails or produces empty output.

    Args:
        pdf_source: The PDF to extract from. Accepts:
            - ``str`` or ``pathlib.Path``: filesystem path to a PDF file.
            - ``io.BytesIO``: in-memory file object (e.g., from Streamlit uploader).
        fallback_to_pypdf: If True (default), fall back to pypdf on any
            pdfplumber failure. Set False only when you need strict pdfplumber.

    Returns:
        A Markdown string with:
        - ``## Heading`` for detected section headings
        - ``- bullet`` for bullet-point items
        - Plain text for body paragraphs
        Returns an empty string if all extraction methods fail.
    """
    # ── Attempt 1: pdfplumber (preferred) ────────────────────────────────────
    try:
        import pdfplumber  # noqa: F401 (import check)
        markdown = _extract_with_pdfplumber(pdf_source)
        if markdown and markdown.strip():
            logger.debug("pdfplumber extraction: %d chars", len(markdown))
            return markdown
        logger.warning("pdfplumber returned empty output — attempting fallback.")
    except ImportError:
        logger.warning(
            "pdfplumber not installed (run: pip install pdfplumber). "
            "Falling back to pypdf."
        )
    except Exception as exc:
        logger.warning("pdfplumber extraction error: %s — attempting fallback.", exc)

    # ── Attempt 2: pypdf fallback ─────────────────────────────────────────────
    if fallback_to_pypdf:
        text = _fallback_extract(pdf_source)
        if text:
            logger.info("pypdf fallback extraction: %d chars", len(text))
        return text

    logger.error("All PDF extraction methods failed for the provided source.")
    return ""


def extract_candidate_metadata(markdown_text: str) -> dict:
    """
    Scrape structured contact and profile metadata from resume Markdown text.

    Uses regular expressions to extract key identifiers that the embedding
    module and (in Phase 2) the communication module will need. All fields
    default to empty string / -1.0 if not found.

    Args:
        markdown_text: Markdown string from ``extract_markdown_from_pdf()``.

    Returns:
        Dict with the following keys:
        - ``email``          : First email address found in the text.
        - ``phone``          : First phone number found (international/local formats).
        - ``linkedin``       : LinkedIn profile URL.
        - ``github``         : GitHub profile URL.
        - ``portfolio``      : Personal website / portfolio URL (non-LinkedIn/GitHub).
        - ``cgpa``           : GPA/CGPA as float on 4.0 scale, or -1.0 if not found.
        - ``skills_section`` : Raw text of the Skills section (for cross-validation).
    """
    result = {
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "cgpa": -1.0,
        "skills_section": "",
    }

    # Email — standard RFC 5321 simplified pattern
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", markdown_text)
    if m:
        result["email"] = m.group(0)

    # Phone — matches international (+92-300-...) and local formats
    m = re.search(r"(\+?[\d][\d\s\-()\.]{7,}\d)", markdown_text)
    if m:
        result["phone"] = m.group(0).strip()

    # LinkedIn URL
    m = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?",
        markdown_text, re.IGNORECASE,
    )
    if m:
        result["linkedin"] = m.group(0).rstrip("/")

    # GitHub URL
    m = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?",
        markdown_text, re.IGNORECASE,
    )
    if m:
        result["github"] = m.group(0).rstrip("/")

    # Portfolio / personal website — any http(s) URL that is NOT LinkedIn, GitHub,
    # or common non-personal domains (drive, docs, mail, maps, ...)
    _skip_domains = (
        "linkedin.com", "github.com", "google.com", "gmail.com",
        "drive.google", "docs.google", "mailto", "facebook.com",
        "twitter.com", "instagram.com", "youtube.com",
    )
    for url_m in re.finditer(
        r"https?://[\w\-./?=#&%+]+",
        markdown_text, re.IGNORECASE,
    ):
        url = url_m.group(0)
        if not any(d in url.lower() for d in _skip_domains):
            result["portfolio"] = url.rstrip("/.,;)]")  # strip trailing punctuation
            break

    # CGPA / GPA extraction — handles common resume formats including:
    #   "3.8/4.0"   "3.67/4.00"  "CGPA- 3.67/4.00"
    #   "GPA: 3.75"  "CGPA: 3.9"  "CGPA of 3.85"  "3.92 out of 4"
    #   "CGPA- 3.67" (dash separator, common in South Asian CVs)
    _cgpa_patterns = [
        # "CGPA- 3.67/4.00" or "GPA- 3.8/4.0" — dash separator (most specific first)
        r"(?:c?gpa)[:\-\s]+(\d\.\d{1,3})\s*/\s*4(?:\.\d{1,2})?",
        # "3.8/4.0" or "3.67/4.00" — standalone fraction
        r"(\d\.\d{1,3})\s*/\s*4(?:\.\d{1,2})?",
        # "CGPA: 3.85" or "GPA: 3.85" or "CGPA of 3.85" — colon/space separator
        r"(?:c?gpa|grade\s+point\s+average)[:\-\s]+(?:of\s+)?(\d\.\d{1,3})",
        # "3.92 out of 4"
        r"(\d\.\d{1,3})\s+out\s+of\s+4",
    ]
    for pat in _cgpa_patterns:
        m = re.search(pat, markdown_text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 4.0:          # Sanity-check: must be on 4.0 scale
                    result["cgpa"] = round(val, 2)
                    break
            except (ValueError, IndexError):
                pass

    # Skills section — capture content between "Skills" heading and next heading
    m = re.search(
        r"(?:##?\s+)?(?:technical\s+)?skills[:\s]*\n(.*?)(?=\n##|\Z)",
        markdown_text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        result["skills_section"] = m.group(1).strip()[:500]  # Cap at 500 chars

    return result


# ── pdfplumber Core Extraction ────────────────────────────────────────────────

def _extract_with_pdfplumber(pdf_source: PDFSource) -> str:
    """
    Layout-aware text extraction using pdfplumber.

    Per-page algorithm:
    1. Extract all words with bounding boxes (x0, top, x1, bottom) and font sizes.
    2. Detect two-column layout via X-coordinate bimodality heuristic.
    3. Sort words into correct reading order (left column → right column for 2-col).
    4. Group words into logical lines using Y-proximity thresholding.
    5. Classify each line as: Section Heading / Bullet Point / Body Text.
    6. Emit the appropriate Markdown syntax for each line type.

    Args:
        pdf_source: Filesystem path (str/Path) or BytesIO of the PDF.

    Returns:
        Multi-page Markdown string. Pages are separated by ``\\n\\n``.
    """
    import pdfplumber

    page_markdowns: list[str] = []

    with pdfplumber.open(pdf_source) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                page_md = _process_page(page)
                if page_md.strip():
                    page_markdowns.append(page_md)
            except Exception as exc:
                logger.warning("Error processing page %d: %s — skipped.", page_num, exc)

    return "\n\n".join(page_markdowns)


def _process_page(page) -> str:
    """
    Extract and structure text from a single pdfplumber Page object.

    Handles multi-column layouts by detecting the X-coordinate distribution
    of words and sorting them into columns before line-grouping.

    Args:
        page: A ``pdfplumber.Page`` instance.

    Returns:
        Markdown string for this page. Empty string if the page has no text.
    """
    # Extract words with bounding boxes.
    # extra_attrs=["size"] requests character-level font size information.
    # x_tolerance and y_tolerance control when adjacent characters are merged.
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,   # We manage reading order ourselves
        extra_attrs=["size"],
    )

    if not words:
        return ""

    page_width = float(page.width)
    page_mid = page_width / 2.0

    # ── Two-column detection ──────────────────────────────────────────────────
    # Heuristic: if significant word mass is present in BOTH the left and right
    # halves of the page, treat it as a two-column layout.
    # The 1.1 / 0.9 multipliers create a small overlap zone at the midpoint to
    # avoid mis-classifying words that straddle the centre line.
    left_words  = [w for w in words if w["x1"] <= page_mid * 1.1]
    right_words = [w for w in words if w["x0"] >= page_mid * 0.9]

    # Require both halves to have at least 10% of total words to call it two-column
    min_column_fraction = 0.10
    is_two_column = (
        len(left_words) / max(len(words), 1) >= min_column_fraction
        and len(right_words) / max(len(words), 1) >= min_column_fraction
        and len(left_words) > 5
        and len(right_words) > 5
    )

    # ── Reading-order sort ────────────────────────────────────────────────────
    if is_two_column:
        # Read left column top-to-bottom, then right column top-to-bottom
        left_sorted  = sorted(left_words,  key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        right_sorted = sorted(right_words, key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        ordered_words = left_sorted + right_sorted
    else:
        # Standard left-to-right, top-to-bottom sort
        ordered_words = sorted(words, key=lambda w: (round(w["top"] / 3) * 3, w["x0"]))

    # ── Group into logical text lines ─────────────────────────────────────────
    lines = _group_words_into_lines(ordered_words, y_tolerance=5)

    # ── Compute page-level average font size for heading detection ────────────
    sizes = [w.get("size", 10.0) for w in words if w.get("size")]
    avg_size = sum(sizes) / len(sizes) if sizes else 10.0

    # ── Convert each line to Markdown ─────────────────────────────────────────
    markdown_lines: list[str] = []

    for line_words in lines:
        line_text = " ".join(w["text"] for w in line_words).strip()
        if not line_text:
            continue

        # Compute average font size for THIS line
        line_sizes = [w.get("size", avg_size) for w in line_words if w.get("size")]
        line_avg_size = sum(line_sizes) / len(line_sizes) if line_sizes else avg_size

        # ── Heading score (higher = more likely a section heading) ────────────
        heading_score = 0

        if line_text.isupper() and len(line_text) > 2:
            heading_score += 2   # ALL CAPS is the strongest visual cue

        if line_avg_size > avg_size * 1.12:
            heading_score += 2   # Larger-than-average font

        if len(line_text.split()) <= 6:
            heading_score += 1   # Short lines are likely headings

        if not any(c in line_text for c in ".,;:()"):
            heading_score += 1   # Section names rarely have punctuation

        text_lower = line_text.lower()
        if any(kw in text_lower for kw in _HEADING_KEYWORDS):
            heading_score += 2   # Matches a known section keyword

        # ── Bullet detection ──────────────────────────────────────────────────
        bullet_chars = set("•-*◦–▪●○▶►→")
        is_bullet = line_text[0] in bullet_chars

        # ── Emit Markdown ─────────────────────────────────────────────────────
        if heading_score >= 4:
            # Normalise to Title Case and wrap as Markdown heading
            clean = line_text.strip("".join(bullet_chars) + " ").strip()
            markdown_lines.append(f"\n## {clean.title()}")

        elif is_bullet:
            # Strip the bullet character and emit as a Markdown list item
            clean = re.sub(r"^[•\-\*◦–▪●○▶►→]\s*", "", line_text).strip()
            markdown_lines.append(f"- {clean}")

        else:
            markdown_lines.append(line_text)

    return "\n".join(markdown_lines)


def _group_words_into_lines(words: list[dict], y_tolerance: int = 5) -> list[list[dict]]:
    """
    Group word dicts into logical text lines by Y-coordinate proximity.

    Words whose ``top`` values differ by no more than ``y_tolerance`` pixels
    are considered to belong to the same horizontal line of text. This threshold
    accounts for minor baseline variations within a single visual line.

    Args:
        words: List of word dicts from pdfplumber (must contain the ``top`` key).
        y_tolerance: Maximum pixel difference in top-position to treat two
                     words as being on the same line. Default 5px.

    Returns:
        List of lines, where each line is a list of word dicts sorted by X.
    """
    if not words:
        return []

    lines: list[list[dict]] = []
    current_line: list[dict] = [words[0]]
    current_y = words[0]["top"]

    for word in words[1:]:
        word_y = word["top"]
        if abs(word_y - current_y) <= y_tolerance:
            current_line.append(word)
        else:
            # Sort the completed line by X-position for correct L→R ordering
            lines.append(sorted(current_line, key=lambda w: w["x0"]))
            current_line = [word]
            current_y = word_y

    # Flush the last line
    if current_line:
        lines.append(sorted(current_line, key=lambda w: w["x0"]))

    return lines


# ── pypdf Fallback ─────────────────────────────────────────────────────────────

def _fallback_extract(pdf_source: PDFSource) -> str:
    """
    Simple flat text extraction using pypdf as a last-resort fallback.

    This method does NOT preserve multi-column layout. Text may be in a
    degraded, mixed-up order for complex resume layouts. A comment marker
    is prepended to the output so downstream code or logs can distinguish
    pypdf output from pdfplumber output.

    Args:
        pdf_source: Filesystem path (str/Path) or BytesIO of the PDF.

    Returns:
        Flat plain text (not Markdown), prefixed with an HTML comment marker.
        Empty string if pypdf also fails.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error(
            "Neither pdfplumber nor pypdf is installed. "
            "Cannot extract text from PDF. "
            "Run: pip install pdfplumber pypdf"
        )
        return ""

    try:
        # Accept both file paths and in-memory BytesIO objects
        if isinstance(pdf_source, (str, Path)):
            reader = PdfReader(str(pdf_source))
        else:
            pdf_source.seek(0)   # Rewind to start for re-reads
            reader = PdfReader(pdf_source)

        pages: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    pages.append(text)
            except Exception as page_exc:
                logger.debug("pypdf: skipping corrupt page — %s", page_exc)

        if not pages:
            logger.warning("pypdf extracted no text from this PDF.")
            return ""

        flat = "\n".join(pages)
        logger.debug("pypdf fallback: extracted %d chars.", len(flat))
        # Prefix so caller can detect degraded extraction quality
        return f"<!-- extracted-via: pypdf-fallback -->\n{flat}"

    except Exception as exc:
        logger.error("pypdf extraction failed: %s", exc)
        return ""
