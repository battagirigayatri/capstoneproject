import json
import fitz
import os


def extract_text_elements(pdf_path):
    doc = fitz.open(pdf_path)
    elements = []

    for page_num, page in enumerate(doc, start=1):

        # Extract normal page text
        text = page.get_text("text").strip()

        if text:
            elements.append({
                "content": text,
                "source_page": page_num,
                "element_type": "text",
                "confidence_score": 0.95
            })

    doc.close()
    return elements


def combine_all():

    print("Reading PDF...")

    text_elements = extract_text_elements("pdfs/sample_paper.pdf")

    print(f"Text elements found: {len(text_elements)}")

    # Load tables if available
    if os.path.exists("output_tables.json"):
        with open("output_tables.json", "r", encoding="utf-8") as f:
            table_elements = json.load(f)
    else:
        table_elements = []

    # Load formulas if available
    if os.path.exists("output_formulas.json"):
        with open("output_formulas.json", "r", encoding="utf-8") as f:
            formula_elements = json.load(f)
    else:
        formula_elements = []

    combined = text_elements + table_elements + formula_elements

    with open(
        "output_combined_structured.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print()
    print("Combined document created successfully!")
    print(f"Text elements: {len(text_elements)}")
    print(f"Table elements: {len(table_elements)}")
    print(f"Formula elements: {len(formula_elements)}")
    print(f"TOTAL elements: {len(combined)}")
    print("Output: output_combined_structured.json")


if __name__ == "__main__":
    combine_all()