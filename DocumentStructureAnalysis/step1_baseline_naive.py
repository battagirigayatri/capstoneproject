"""
STEP 1 (multi-PDF): Baseline Naive Extraction
----------------------------------------------
Reads EVERY pdf inside the 'pdfs' folder and extracts plain text with no
layout awareness. This is your "before" evidence.

Outputs:
  output_naive.json  -> used by step 8 (doc, page, text)
  output_naive.txt   -> readable file to show the problem in your report
"""

import glob
import json
import os
import fitz  # PyMuPDF  (if you use "import pymupdf" in your setup, change this line)

PDF_DIR = "pdfs"


def naive_extract_all(pdf_dir=PDF_DIR):
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        raise SystemExit(f"No PDFs found. Put your 3 PDFs inside the '{pdf_dir}' folder.")

    records, txt = [], ""
    for path in pdf_files:
        name = os.path.basename(path)
        doc = fitz.open(path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()  # naive: no layout awareness
            records.append({"doc": name, "source_page": page_num, "text": text})
            txt += f"\n=== {name} | PAGE {page_num} ===\n{text}"
        print(f"{name}: {len(doc)} pages")
        doc.close()

    with open("output_naive.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open("output_naive.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\nNaive extraction done for {len(pdf_files)} PDFs -> output_naive.json, output_naive.txt")


if __name__ == "__main__":
    naive_extract_all()
