"""
STEP 2: Fix Reading Order (Multi-Column PDFs)
-----------------------------------------------
Uses PyMuPDF block coordinates to detect columns and reorder text
so multi-column pages read top-to-bottom, left-column-then-right.
"""

import fitz  # PyMuPDF


def get_blocks_with_position(page):
    """Extract text blocks with their bounding box coordinates."""
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    # Keep only text blocks (block_type == 0), skip images
    return [b for b in blocks if b[6] == 0 and b[4].strip()]


def detect_columns(blocks, page_width):
    """
    Simple column detection: split blocks into left/right half
    based on x0 coordinate relative to page width midpoint.
    Works for single or two-column layouts.
    """
    midpoint = page_width / 2
    left_col = [b for b in blocks if b[0] < midpoint]
    right_col = [b for b in blocks if b[0] >= midpoint]

    # If almost everything is in one column, treat as single-column page
    if len(right_col) < 0.15 * len(blocks):
        return [blocks], False  # single column
    return [left_col, right_col], True  # two columns


def sort_blocks_reading_order(blocks):
    """Sort top-to-bottom, then left-to-right within same line band."""
    return sorted(blocks, key=lambda b: (round(b[1], 0), b[0]))


def extract_in_reading_order(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        blocks = get_blocks_with_position(page)
        columns, is_multicol = detect_columns(blocks, page.rect.width)

        page_text = f"\n--- PAGE {page_num} ({'2-column' if is_multicol else '1-column'}) ---\n"
        for col_blocks in columns:
            ordered = sort_blocks_reading_order(col_blocks)
            for b in ordered:
                page_text += b[4] + "\n"

        full_text += page_text

    doc.close()

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Reading-order-corrected extraction done: {output_txt_path}")

if __name__ == "__main__":
    extract_in_reading_order("pdfs/sample_paper.pdf", "output_reading_order_fixed.txt")
