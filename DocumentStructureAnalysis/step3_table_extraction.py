"""
STEP 3: Table Extraction (Structured, Not Flattened)
-------------------------------------------------------
Uses pdfplumber to detect tables and keep them as row/column
structured data instead of jumbled text. Falls back gracefully
if a page has no tables.
"""

import pdfplumber
import json


def extract_tables(pdf_path, output_json_path):
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                # Clean None cells
                cleaned_table = [
                    [cell if cell is not None else "" for cell in row]
                    for row in table
                ]
                all_tables.append({
                    "source_page": page_num,
                    "table_index_on_page": t_idx,
                    "rows": cleaned_table,
                    "num_rows": len(cleaned_table),
                    "num_cols": len(cleaned_table[0]) if cleaned_table else 0,
                    "element_type": "table",
                    "confidence_score": estimate_table_confidence(cleaned_table),
                })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(all_tables)} tables -> {output_json_path}")
    return all_tables


def estimate_table_confidence(table):
    """
    Simple heuristic confidence score (0 to 1):
    Based on how many cells are non-empty (fewer blanks = higher confidence
    that structure was captured correctly).
    """
    if not table:
        return 0.0
    total_cells = sum(len(row) for row in table)
    if total_cells == 0:
        return 0.0
    filled_cells = sum(1 for row in table for cell in row if cell.strip())
    return round(filled_cells / total_cells, 2)


if __name__ == "__main__":
    extract_tables("pdfs/sample_paper.pdf", "output_tables.json")
