"""Debug: simulate what the pipeline does and check key matching."""
import sys
sys.path.insert(0, ".")
from modules.parser import extract_markdown_from_pdf, extract_candidate_metadata
from pathlib import Path

folder = Path("test_cvs")
filenames = []
for pdf in sorted(folder.glob("*.pdf")):
    fn = pdf.name
    display_name = Path(fn).stem.replace("_", " ").replace("-", " ").title()
    filenames.append((fn, display_name))
    print(f"filename: {fn!r}")
    print(f"  display_name (CandidateRecord.name): {display_name!r}")
    print(f"  cand_name from results_df row would be: (same as display_name)")
    print()
