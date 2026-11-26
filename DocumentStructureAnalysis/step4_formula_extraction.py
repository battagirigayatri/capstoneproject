"""
STEP 4: Formula Extraction (Using Nougat)
--------------------------------------------
Nougat (Meta's OCR model for academic PDFs) converts pages into
markdown/LaTeX, which preserves formulas instead of dropping them.

Install: pip install nougat-ocr

NOTE: Nougat runs as a command-line tool on the whole PDF. This
script wraps it and then pulls out only the formula lines
(anything wrapped in $...$ or $$...$$) as separate elements.
"""

import subprocess
import re
import json
import os


def run_nougat(pdf_path, output_dir="nougat_output"):
    """Runs Nougat OCR on the PDF, producing a .mmd (markdown) file."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        ["nougat", pdf_path, "-o", output_dir],
        check=True
    )
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(output_dir, f"{base_name}.mmd")


def extract_formulas_from_mmd(mmd_path, output_json_path):
    with open(mmd_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match $$...$$ (block formulas) and $...$ (inline formulas)
    block_formulas = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)
    inline_formulas = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", content)

    formulas = []
    for f_text in block_formulas:
        formulas.append({
            "formula_latex": f_text.strip(),
            "type": "block",
            "element_type": "formula",
            "confidence_score": 0.9,  # Nougat block formulas tend to be reliable
        })
    for f_text in inline_formulas:
        formulas.append({
            "formula_latex": f_text.strip(),
            "type": "inline",
            "element_type": "formula",
            "confidence_score": 0.75,
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(formulas)} formulas -> {output_json_path}")
    return formulas


if __name__ == "__main__":
    mmd_file = run_nougat("pdfs/sample.pdf")
    extract_formulas_from_mmd(mmd_file, "output_formulas.json")
