"""Full pipeline simulation — checks every step from PDF → CandidateRecord → lookup."""
import sys
sys.path.insert(0, ".")
from modules.parser import extract_markdown_from_pdf, extract_candidate_metadata
from modules.ingestion import CandidateRecord
from pathlib import Path

folder = Path("test_cvs")
pdf_files = sorted(folder.glob("*.pdf"))

filenames = []
resume_markdowns = []
candidates = []
candidate_metadata = []

print("=== STEP 1: Building CandidateRecords ===")
for pdf_path in pdf_files:
    fn = pdf_path.name
    md = extract_markdown_from_pdf(pdf_path)
    meta = extract_candidate_metadata(md)
    display_name = Path(fn).stem.replace("_", " ").replace("-", " ").title()

    rec = CandidateRecord(
        name=display_name,
        email=meta.get("email", ""),
        phone=meta.get("phone", ""),
        linkedin_url=meta.get("linkedin", ""),
        cgpa=meta.get("cgpa", -1.0),
        resume_markdown=md,
    )
    rec.job_title  = meta.get("github", "")
    rec.cover_note = meta.get("portfolio", "")

    candidates.append(rec)
    candidate_metadata.append({
        "name":      display_name,
        "email":     rec.email,
        "cgpa":      rec.cgpa,
        "years_exp": rec.years_exp,
        "degree":    rec.degree,
    })
    filenames.append(fn)
    resume_markdowns.append(md)

    print(f"  File: {fn}")
    print(f"    display_name: {display_name!r}")
    print(f"    rec.cgpa:     {rec.cgpa}")
    print(f"    meta['cgpa']: {meta['cgpa']}")
    print()

print("\n=== STEP 2: record_by_name lookup keys ===")
record_by_name = {c.name: c for c in candidates}
for k, v in record_by_name.items():
    print(f"  key={k!r}  cgpa={v.cgpa}")

print("\n=== STEP 3: Simulating results_df row lookup ===")
# Simulate what the results_df 'Candidate Name' column would contain
for cm in candidate_metadata:
    cand_name = cm["name"]
    record = record_by_name.get(cand_name)
    if record:
        print(f"  cand_name={cand_name!r}")
        print(f"    -> Found record, cgpa={record.cgpa}")
        print(f"    -> cgpa badge shown: {record.cgpa != -1.0}")
    else:
        print(f"  cand_name={cand_name!r} -> RECORD NOT FOUND!")
