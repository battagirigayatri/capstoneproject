
"""
STEP 2: Multi-PDF Structured Extraction

Features:
1. Reading-order correction for multi-column PDFs.
2. Table extraction using pdfplumber.
3. Formula detection using mathematical symbols and fonts.
4. Text extraction and chunking.
5. Confidence score assignment.
"""

import glob
import json
import os
import re

import fitz
import pdfplumber


PDF_DIR = "pdfs"
CHUNK_CHARS = 500

MATH_CHARS = set(
    "=+−*/^_∑∫√≤≥≈≠∂∆∇αβγδθλμπσωΣΠ±×÷"
)

MATH_FONT_RE = re.compile(
    r"(cmmi|cmsy|cmex|msam|msbm|mathematicalpi|"
    r"cambria ?math|stix|mtmi|mtsy|mtex|"
    r"txmi|txsy|lmmath|latinmodern-math|eufm|rsfs|symbol)",
    re.I
)


# -------------------------------------------------
# READING ORDER
# -------------------------------------------------

def reading_order(blocks, page_w):

    full = sorted(
        [b for b in blocks if (b[2] - b[0]) > 0.6 * page_w],
        key=lambda b: b[1]
    )

    others = [b for b in blocks if b not in full]

    mid = page_w / 2

    bands = [[] for _ in range(len(full) + 1)]

    for b in others:
        idx = sum(1 for f in full if f[1] <= b[1])
        bands[idx].append(b)

    def sort_band(items):

        left = sorted(
            [b for b in items if (b[0] + b[2]) / 2 < mid],
            key=lambda b: (b[1], b[0])
        )

        right = sorted(
            [b for b in items if (b[0] + b[2]) / 2 >= mid],
            key=lambda b: (b[1], b[0])
        )

        if len(left) < 2 or len(right) < 2:
            return sorted(items, key=lambda b: (b[1], b[0]))

        return left + right

    ordered = []

    for i, band in enumerate(bands):

        ordered += sort_band(band)

        if i < len(full):
            ordered.append(full[i])

    return ordered


def is_two_column(blocks, page_w):

    mid = page_w / 2

    narrow = [
        b for b in blocks
        if (b[2] - b[0]) <= 0.6 * page_w
    ]

    left = [
        b for b in narrow
        if (b[0] + b[2]) / 2 < mid
    ]

    right = [
        b for b in narrow
        if (b[0] + b[2]) / 2 >= mid
    ]

    return len(left) >= 2 and len(right) >= 2


# -------------------------------------------------
# CONFIDENCE HELPERS
# -------------------------------------------------

def text_confidence(text):

    if len(text) < 3:
        return 0.3

    weird = sum(
        1 for c in text
        if not (
            c.isalnum()
            or c.isspace()
            or c in ".,;:!?()-'\"%/"
        )
    )

    return round(max(0.3, 1 - 3 * weird / len(text)), 2)


def table_confidence(rows):

    cells = [c for r in rows for c in r]

    if not cells:
        return 0.0

    return round(
        sum(1 for c in cells if c.strip()) / len(cells),
        2
    )


def math_font_ratio(dict_block):

    if not dict_block:
        return 0.0

    total = 0
    math_chars = 0

    for line in dict_block.get("lines", []):

        for span in line.get("spans", []):

            n = len(span["text"].strip())

            total += n

            if MATH_FONT_RE.search(span["font"]):
                math_chars += n

    return math_chars / total if total else 0.0


def looks_like_formula(text, font_ratio=0.0):

    t = text.strip()

    if not t or len(t) > 250:
        return False, 0.0

    long_words = len(re.findall(r"[A-Za-z]{5,}", t))

    symbol_rule = (
        any(c in MATH_CHARS for c in t)
        and long_words <= 2
    )

    font_rule = (
        font_ratio >= 0.3
        and long_words <= 4
    )

    if font_rule:
        return True, 0.8

    if symbol_rule:
        return True, 0.6

    return False, 0.0


# -------------------------------------------------
# FORMULA MERGING
# -------------------------------------------------

def merge_formula_pieces(pieces):

    merged = []

    for text, bbox in pieces:

        if merged:

            pt, pb = merged[-1]

            near_y = (
                bbox[1] <= pb[3] + 4
                and bbox[3] >= pb[1] - 4
            )

            near_x = (
                bbox[0] <= pb[2] + 40
                and bbox[2] >= pb[0] - 40
            )

            if near_y and near_x:

                merged[-1] = (
                    pt + " " + text,
                    (
                        min(pb[0], bbox[0]),
                        min(pb[1], bbox[1]),
                        max(pb[2], bbox[2]),
                        max(pb[3], bbox[3])
                    )
                )

                continue

        merged.append((text, bbox))

    return merged


# -------------------------------------------------
# TABLE BOUNDARY HELPER
# -------------------------------------------------

def inside(block, bbox):

    cx = (block[0] + block[2]) / 2
    cy = (block[1] + block[3]) / 2

    return (
        bbox[0] <= cx <= bbox[2]
        and bbox[1] <= cy <= bbox[3]
    )


# -------------------------------------------------
# TEXT CHUNKING
# -------------------------------------------------

def chunk_blocks(texts):

    chunks = []
    cur = ""

    for t in texts:

        t = " ".join(t.split())

        if cur and len(cur) + len(t) > CHUNK_CHARS:

            chunks.append(cur)
            cur = ""

        cur = (cur + " " + t).strip()

    if cur:
        chunks.append(cur)

    return chunks


# -------------------------------------------------
# MAIN STRUCTURED EXTRACTION
# -------------------------------------------------

def process_pdf(path):

    name = os.path.basename(path)

    elements = []

    doc = fitz.open(path)
    plumber = pdfplumber.open(path)

    try:

        for page_num, page in enumerate(doc, start=1):

            # -------------------------------------
            # TABLE EXTRACTION
            # -------------------------------------

            table_boxes = []

            try:

                for t in plumber.pages[page_num - 1].find_tables():

                    extracted_rows = t.extract()

                    rows = [
                        [(c or "").strip() for c in r]
                        for r in extracted_rows
                    ]

                    if len(rows) >= 2 and len(rows[0]) >= 2:

                        table_boxes.append(t.bbox)

                        elements.append({
                            "doc": name,
                            "source_page": page_num,
                            "element_type": "table",
                            "rows": rows,
                            "confidence_score": table_confidence(rows)
                        })

            except Exception as e:

                print(
                    f"Table warning {name} "
                    f"page {page_num}: {e}"
                )

            # -------------------------------------
            # TEXT BLOCK EXTRACTION
            # -------------------------------------

            blocks = [
                b for b in page.get_text("blocks")
                if b[6] == 0
                and b[4].strip()
                and not any(inside(b, bb) for bb in table_boxes)
            ]

            ordered = reading_order(
                blocks,
                page.rect.width
            )

            dict_blocks = {
                b["number"]: b
                for b in page.get_text("dict")["blocks"]
                if b.get("type") == 0
            }

            text_parts = []
            formula_pieces = []
            formula_conf = []

            for b in ordered:

                is_f, conf = looks_like_formula(
                    b[4],
                    math_font_ratio(dict_blocks.get(b[5]))
                )

                if is_f:

                    formula_pieces.append(
                        (" ".join(b[4].split()), b[:4])
                    )

                    formula_conf.append(conf)

                else:

                    text_parts.append(b[4])

            # -------------------------------------
            # FORMULA EXTRACTION
            # -------------------------------------

            formula_pieces.sort(
                key=lambda p: (
                    round(p[1][1] / 15),
                    p[1][0]
                )
            )

            conf_value = (
                max(formula_conf)
                if formula_conf
                else 0.6
            )

            for text, _ in merge_formula_pieces(formula_pieces):

                elements.append({
                    "doc": name,
                    "source_page": page_num,
                    "element_type": "formula",
                    "content": text,
                    "formula_latex": text,
                    "confidence_score": conf_value
                })

            # -------------------------------------
            # NORMAL TEXT ELEMENTS
            # -------------------------------------

            for chunk in chunk_blocks(text_parts):

                elements.append({
                    "doc": name,
                    "source_page": page_num,
                    "element_type": "text",
                    "content": chunk,
                    "confidence_score": text_confidence(chunk)
                })

        # -----------------------------------------
        # PAGE LAYOUT DETECTION
        # -----------------------------------------

        layout = "2-column" if any(
            is_two_column(
                [b for b in p.get_text("blocks") if b[6] == 0],
                p.rect.width
            )
            for p in doc
        ) else "1-column"

        n_pages = len(doc)

    finally:

        doc.close()
        plumber.close()

    return elements, n_pages, layout


# -------------------------------------------------
# ELEMENT PREVIEW
# -------------------------------------------------

def element_preview(e):

    if e["element_type"] == "table":

        return (
            "TABLE: "
            + " | ".join(
                " , ".join(r)
                for r in e["rows"][:3]
            )[:140]
        )

    return (
        e["element_type"].upper()
        + ": "
        + e.get("content", "")[:140]
    )


# -------------------------------------------------
# MULTI-PDF EXECUTION
# -------------------------------------------------

if __name__ == "__main__":

    pdf_files = sorted(
        glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    )

    if not pdf_files:

        raise SystemExit(
            f"No PDFs found in '{PDF_DIR}' folder."
        )

    all_elements = []
    helper = ""

    for path in pdf_files:

        elems, n_pages, layout = process_pdf(path)

        all_elements += elems

        counts = {
            t: sum(
                1 for e in elems
                if e["element_type"] == t
            )
            for t in ("text", "table", "formula")
        }

        print(
            f"{os.path.basename(path)}: "
            f"{n_pages} pages, "
            f"layout={layout}, "
            f"text={counts['text']}, "
            f"tables={counts['table']}, "
            f"formulas={counts['formula']}"
        )

        helper += (
            f"\n######## {os.path.basename(path)} "
            f"({layout}) ########\n"
        )

        for e in elems:

            helper += (
                f"[page {e['source_page']}] "
                f"{element_preview(e)}\n"
            )

    with open(
        "output_combined_structured.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_elements,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        "question_helper.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(helper)

    print(
        f"\nTotal elements: {len(all_elements)} "
        "-> output_combined_structured.json"
    )

    print(
        "Open question_helper.txt to see extracted elements."
    )
