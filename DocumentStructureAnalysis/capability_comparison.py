"""
CAPABILITY COMPARISON GRAPH  ->  graph6_capability_comparison.png
-------------------------------------------------------------------
Shows what each pipeline actually recovers from YOUR pdfs.
Reads output_combined_structured.json (made by step2_structured_extract.py).
Naive extraction returns plain text only, so it recovers 0 of these.

Run:  python capability_comparison.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open("output_combined_structured.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

tables = sum(1 for e in elements if e["element_type"] == "table")
formulas = sum(1 for e in elements if e["element_type"] == "formula")
tagged = sum(1 for e in elements if e.get("source_page") and e.get("confidence_score") is not None)

labels = ["Tables kept as\nrows & columns", "Formulas kept as\nseparate elements", "Elements with page\n+ confidence score"]
naive = [0, 0, 0]
structured = [tables, formulas, tagged]

x = range(len(labels))
plt.figure(figsize=(8, 4.8))
plt.bar([i - 0.2 for i in x], naive, 0.4, label="Naive (plain text)", color="#d9534f")
plt.bar([i + 0.2 for i in x], structured, 0.4, label="Structured (ours)", color="#2e7d32")
top = max(structured + [1])
for i in x:
    plt.text(i - 0.2, naive[i] + top * 0.015, str(naive[i]), ha="center", fontsize=10)
    plt.text(i + 0.2, structured[i] + top * 0.015, str(structured[i]), ha="center", fontsize=10)
plt.xticks(list(x), labels)
plt.ylabel("Number of items recovered")
plt.ylim(0, top * 1.15)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title("What Each Pipeline Recovers from the Test Documents")
plt.legend()
plt.tight_layout()
plt.savefig("graph6_capability_comparison.png", dpi=200)
plt.close()

print(f"Tables: {tables}   Formulas: {formulas}   Tagged elements: {tagged}")
print("Saved -> graph6_capability_comparison.png")
