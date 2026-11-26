"""
STEP 8 (multi-PDF): Evaluate Naive vs Structured Retrieval + Graphs
--------------------------------------------------------------------
Needs (same folder):
  output_naive.json                (step 1)
  output_combined_structured.json  (step 2)
  test_questions.json              (you write it, format below)

A retrieval counts as correct only if BOTH the document and the page match.

test_questions.json format:
[
  {"question": "...", "doc": "paper1.pdf", "correct_page": 2, "type": "text"},
  {"question": "...", "doc": "report.pdf", "correct_page": 4, "type": "table"},
  {"question": "...", "doc": "paper2.pdf", "correct_page": 1, "type": "formula"}
]
type = text / table / formula

Install once:  pip install scikit-learn matplotlib
"""

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 3
NAIVE_CHUNK_CHARS = 500


# ---------------- load ----------------
def load_naive():
    with open("output_naive.json", "r", encoding="utf-8") as f:
        pages = json.load(f)
    texts, keys = [], []
    for p in pages:
        t = " ".join(p["text"].split())
        for i in range(0, max(len(t), 1), NAIVE_CHUNK_CHARS):
            chunk = t[i:i + NAIVE_CHUNK_CHARS]
            if len(chunk) > 20:
                texts.append(chunk)
                keys.append((p["doc"], p["source_page"]))
    return texts, keys


def element_text(e):
    if e["element_type"] == "table":
        return "table " + " ".join(" ".join(r) for r in e["rows"])
    return e["content"]


def load_structured():
    with open("output_combined_structured.json", "r", encoding="utf-8") as f:
        elements = json.load(f)
    return elements, [element_text(e) for e in elements], [(e["doc"], e["source_page"]) for e in elements]


# ---------------- evaluation ----------------
def evaluate(questions, texts, keys):
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
    matrix = vec.fit_transform(texts)
    hit1 = hit3 = 0
    by_type = defaultdict(lambda: [0, 0])
    by_doc = defaultdict(lambda: [0, 0])
    misses = []
    for q in questions:
        scores = cosine_similarity(vec.transform([q["question"]]), matrix)[0]
        top = [keys[i] for i in scores.argsort()[::-1][:TOP_K]]
        target = (q["doc"], q["correct_page"])
        ok1, ok3 = top[0] == target, target in top
        hit1 += ok1
        hit3 += ok3
        by_type[q.get("type", "text")][0] += ok3
        by_type[q.get("type", "text")][1] += 1
        by_doc[q["doc"]][0] += ok3
        by_doc[q["doc"]][1] += 1
        if not ok3:
            misses.append(q["question"])
    n = len(questions)
    pct = lambda h, c: round(100 * h / c, 1)
    return {
        "recall@1": pct(hit1, n), "recall@3": pct(hit3, n),
        "by_type": {t: pct(h, c) for t, (h, c) in by_type.items()},
        "by_doc": {d: pct(h, c) for d, (h, c) in by_doc.items()},
        "missed_questions": misses,
    }


# ---------------- graphs ----------------
def grouped_bar(labels, naive_vals, struct_vals, title, ylabel, filename):
    x = range(len(labels))
    plt.figure(figsize=(max(6, 1.4 * len(labels) + 3), 4.2))
    plt.bar([i - 0.2 for i in x], naive_vals, 0.4, label="Naive", color="#d9534f")
    plt.bar([i + 0.2 for i in x], struct_vals, 0.4, label="Structured (ours)", color="#2e7d32")
    for i in x:
        plt.text(i - 0.2, naive_vals[i] + 1, f"{naive_vals[i]:g}", ha="center", fontsize=8)
        plt.text(i + 0.2, struct_vals[i] + 1, f"{struct_vals[i]:g}", ha="center", fontsize=8)
    plt.xticks(list(x), labels, rotation=15 if len(labels) > 3 else 0)
    plt.ylabel(ylabel)
    plt.ylim(0, 112)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_all(naive, struct, elements):
    grouped_bar(["Recall@1", "Recall@3"], [naive["recall@1"], naive["recall@3"]],
                [struct["recall@1"], struct["recall@3"]],
                "Naive vs Structured Pipeline", "Retrieval accuracy (%)", "graph1_naive_vs_structured.png")

    types = sorted(set(naive["by_type"]) | set(struct["by_type"]))
    grouped_bar(types, [naive["by_type"].get(t, 0) for t in types],
                [struct["by_type"].get(t, 0) for t in types],
                "Accuracy by Element Type", "Recall@3 (%)", "graph2_accuracy_by_type.png")

    docs = sorted(set(naive["by_doc"]) | set(struct["by_doc"]))
    grouped_bar(docs, [naive["by_doc"].get(d, 0) for d in docs],
                [struct["by_doc"].get(d, 0) for d in docs],
                "Accuracy by Document", "Recall@3 (%)", "graph3_accuracy_by_document.png")

    counts, conf = defaultdict(int), defaultdict(list)
    for e in elements:
        counts[e["element_type"]] += 1
        conf[e["element_type"]].append(e["confidence_score"])
    plt.figure(figsize=(5, 4))
    plt.bar(list(counts.keys()), list(counts.values()), color="#1f4e8c")
    plt.ylabel("Number of elements")
    plt.title("Extracted Elements by Type")
    plt.tight_layout()
    plt.savefig("graph4_element_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for t, vals in conf.items():
        plt.hist(vals, bins=10, range=(0, 1), alpha=0.6, label=t)
    plt.xlabel("Confidence score")
    plt.ylabel("Number of elements")
    plt.title("Confidence Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graph5_confidence_distribution.png", dpi=200)
    plt.close()


# ---------------- main ----------------
if __name__ == "__main__":
    with open("test_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    n_texts, n_keys = load_naive()
    elements, s_texts, s_keys = load_structured()

    naive = evaluate(questions, n_texts, n_keys)
    struct = evaluate(questions, s_texts, s_keys)
    plot_all(naive, struct, elements)

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"questions": len(questions), "naive": naive, "structured": struct}, f, indent=2)

    print(f"\nQuestions: {len(questions)}   Documents: {len(struct['by_doc'])}")
    print(f"Naive      Recall@1 {naive['recall@1']}%   Recall@3 {naive['recall@3']}%")
    print(f"Structured Recall@1 {struct['recall@1']}%   Recall@3 {struct['recall@3']}%")
    print("\nRecall@3 by type     naive:", naive["by_type"], " structured:", struct["by_type"])
    print("Recall@3 by document naive:", naive["by_doc"], " structured:", struct["by_doc"])
    print("\nQuestions the structured pipeline missed:")
    for m in struct["missed_questions"] or ["(none)"]:
        print("  -", m)
    print("\nSaved graph1..graph5 PNG files + evaluation_results.json")
