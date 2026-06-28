"""Debug script: shows what text the parser extracts around CGPA/GPA lines."""
import sys
sys.path.insert(0, ".")
from modules.parser import extract_markdown_from_pdf, extract_candidate_metadata
from pathlib import Path

folder = Path("test_cvs")
for pdf in sorted(folder.glob("*.pdf")):
    print(f"\n{'='*60}")
    print(f"FILE: {pdf.name}")
    md = extract_markdown_from_pdf(pdf)

    # Print ALL lines that mention academic/grade keywords
    keywords = ["gpa", "cgpa", "grade", "result", "cumulative", "pointer",
                "4.0", "4.00", "3.", "marks", "percentage", "score"]
    hits = [l for l in md.splitlines()
            if any(kw in l.lower() for kw in keywords)]
    print(f"  Lines with academic keywords ({len(hits)} found):")
    for l in hits[:20]:
        print(f"    {repr(l)}")

    # Show raw context around 'gpa' or 'cgpa' if any
    idx = md.lower().find("cgpa")
    if idx == -1:
        idx = md.lower().find("gpa")
    if idx != -1:
        print(f"\n  Raw text around GPA mention (chars {max(0,idx-50)}–{idx+80}):")
        print(f"    {repr(md[max(0, idx-50): idx+80])}")

    meta = extract_candidate_metadata(md)
    print(f"\n  Extracted CGPA: {meta['cgpa']}")
    print(f"  Email: {meta['email']}")
    print(f"  LinkedIn: {meta['linkedin']}")
