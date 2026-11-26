
import os
import re
import json
import uuid

import fitz
import faiss
import numpy as np

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from step2_structured_extract import process_pdf


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


# -------------------------------------------------
# GLOBAL VARIABLES
# -------------------------------------------------

document_elements = []
metadata = []

vectorizer = None
faiss_index = None

current_filename = None
document_counts = {}


# -------------------------------------------------
# STEP 1: NAIVE EXTRACTION
# -------------------------------------------------

def run_step1(pdf_path, output_dir):

    records = []
    readable_text = ""

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):

        text = page.get_text()

        records.append({
            "doc": os.path.basename(pdf_path),
            "source_page": page_num,
            "text": text
        })

        readable_text += (
            f"\n=== {os.path.basename(pdf_path)} "
            f"| PAGE {page_num} ===\n{text}"
        )

    page_count = len(doc)
    doc.close()

    with open(
        os.path.join(output_dir, "output_naive.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(
        os.path.join(output_dir, "output_naive.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(readable_text)

    return page_count


# -------------------------------------------------
# GET SEARCHABLE CONTENT
# -------------------------------------------------

def get_searchable_content(item):

    content = str(item.get("content", "")).strip()

    # Tables do not have a content field.
    # Convert their rows into searchable text.

    if not content and item.get("rows"):

        rows = item["rows"]

        content = "\n".join(
            " | ".join(str(cell) for cell in row)
            for row in rows
        )

    return content.strip()


# -------------------------------------------------
# BUILD TF-IDF + FAISS INDEX
# -------------------------------------------------

def build_faiss_index(elements):

    global vectorizer, faiss_index, metadata

    searchable_elements = []
    texts = []

    for item in elements:

        content = get_searchable_content(item)

        if content:

            searchable_elements.append(item)
            texts.append(content)

    if not texts:

        raise ValueError(
            "No searchable text, tables, or formulas found."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        stop_words=None
    )

    vectors = vectorizer.fit_transform(texts).toarray().astype("float32")

    if vectors.shape[1] == 0:

        raise ValueError("No searchable vocabulary found.")

    faiss.normalize_L2(vectors)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])

    faiss_index.add(vectors)

    metadata = []

    for item, content in zip(searchable_elements, texts):

        metadata.append({

            "content": content,

            "source_page": item.get("source_page"),

            "element_type": item.get("element_type", "unknown"),

            "confidence_score": item.get("confidence_score"),

            "doc": item.get("doc"),

            "rows": item.get("rows")

        })

    return len(metadata), len(vectorizer.vocabulary_)


# -------------------------------------------------
# RETRIEVE RELEVANT ELEMENTS
# -------------------------------------------------

def search_document(question, top_k=3):

    if vectorizer is None or faiss_index is None:
        return []

    query_vector = vectorizer.transform(
        [question]
    ).toarray().astype("float32")

    if not np.any(query_vector):
        return []

    faiss.normalize_L2(query_vector)

    k = min(top_k, faiss_index.ntotal)

    scores, indices = faiss_index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        # Ignore very weak matches
        if score < 0.08:
            continue

        result = metadata[idx].copy()

        result["relevance_score"] = round(float(score), 4)

        results.append(result)

    return results


# -------------------------------------------------
# QUESTION KEYWORD EXTRACTION
# -------------------------------------------------

STOP_WORDS = set(ENGLISH_STOP_WORDS).union({

    "explain",
    "describe",
    "define",
    "discuss",
    "mention",
    "state",
    "give",
    "write",
    "tell",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "please",
    "according",
    "document",
    "pdf"

})


def get_keywords(text):

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


# -------------------------------------------------
# FIND THE MOST RELEVANT SENTENCE
# -------------------------------------------------

def find_best_sentence(content, question):

    keywords = get_keywords(question)

    if not keywords:
        return None

    # Split the content into individual sentences
    sentences = re.split(
        r'(?<=[.!?])\s+|\n+',
        content
    )

    best_sentence = None
    best_score = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 15:
            continue

        sentence_words = get_keywords(sentence)

        if not sentence_words:
            continue

        matching_words = keywords.intersection(sentence_words)

        # Calculate keyword coverage
        coverage = len(matching_words) / len(keywords)

        # Give preference to sentences focused on the question
        precision = len(matching_words) / len(sentence_words)

        score = coverage + (0.25 * precision)

        # Avoid very long sentences
        if len(sentence) > 350:
            score *= 0.7

        if score > best_score:

            best_score = score
            best_sentence = sentence

    if best_score == 0:
        return None

    return best_sentence


# -------------------------------------------------
# FIND THE MOST RELEVANT TABLE ROW
# -------------------------------------------------

def find_best_table_row(rows, question):

    if not rows:
        return None

    keywords = get_keywords(question)

    if not keywords:
        return None

    headers = [
        str(cell).strip()
        for cell in rows[0]
    ]

    best_row = None
    best_score = 0

    for row in rows[1:]:

        row_values = [
            str(cell).strip()
            for cell in row
        ]

        combined_text = " ".join(
            headers + row_values
        )

        row_words = get_keywords(combined_text)

        matching_words = keywords.intersection(row_words)

        if not matching_words:
            continue

        score = len(matching_words) / len(keywords)

        if score > best_score:

            best_score = score
            best_row = row_values

    if best_row is None:
        return None

    # Return the matching row with its column headings
    answer_parts = []

    for header, value in zip(headers, best_row):

        if value:
            answer_parts.append(f"{header}: {value}")

    return " | ".join(answer_parts)


# -------------------------------------------------
# GENERATE A CONCISE ANSWER
# -------------------------------------------------

def generate_answer(question, results):

    if not results:
        return None

    best_answer = None
    best_result = None
    best_score = 0

    for result in results:

        element_type = result["element_type"]

        # TABLE ANSWER
        if element_type == "table":

            answer = find_best_table_row(
                result.get("rows", []),
                question
            )

        # FORMULA ANSWER
        elif element_type == "formula":

            answer = result.get("content", "").strip()

        # TEXT ANSWER
        else:

            answer = find_best_sentence(
                result.get("content", ""),
                question
            )

        if not answer:
            continue

        # Combine retrieval relevance with keyword matching
        answer_words = get_keywords(answer)
        question_words = get_keywords(question)

        if not question_words:
            continue

        overlap = len(
            answer_words.intersection(question_words)
        ) / len(question_words)

        final_score = (
            0.6 * overlap
            + 0.4 * result["relevance_score"]
        )

        if final_score > best_score:

            best_score = final_score
            best_answer = answer
            best_result = result

    if best_answer is None:
        return None

    # Return only one concise answer
    return {

        "content": best_answer,

        "source_page": best_result["source_page"],

        "element_type": best_result["element_type"],

        "confidence_score": best_result["confidence_score"],

        "relevance_score": best_result["relevance_score"],

        "doc": best_result["doc"]

    }


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return render_template(

        "index.html",

        message=None,

        filename=current_filename,

        counts=document_counts,

        results=None,

        question=""

    )


# -------------------------------------------------
# PROCESS UPLOADED PDF
# -------------------------------------------------

@app.route("/process", methods=["POST"])
def process_document():

    global document_elements
    global current_filename
    global document_counts
    global vectorizer
    global faiss_index
    global metadata

    pdf_file = request.files.get("pdf")

    if not pdf_file or not pdf_file.filename:

        return render_template(
            "index.html",
            message="Please select a PDF file.",
            filename=current_filename,
            counts=document_counts,
            results=None,
            question=""
        )

    if not pdf_file.filename.lower().endswith(".pdf"):

        return render_template(
            "index.html",
            message="Only PDF files are supported.",
            filename=current_filename,
            counts=document_counts,
            results=None,
            question=""
        )

    safe_name = secure_filename(pdf_file.filename)

    document_id = uuid.uuid4().hex

    output_dir = os.path.join(
        app.config["UPLOAD_FOLDER"],
        document_id
    )

    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, safe_name)

    pdf_file.save(pdf_path)

    try:

        # STEP 1: NAIVE EXTRACTION
        page_count = run_step1(
            pdf_path,
            output_dir
        )

        # STEP 2: STRUCTURED EXTRACTION
        elements, structured_pages, layout = process_pdf(
            pdf_path
        )

        structured_path = os.path.join(
            output_dir,
            "output_combined_structured.json"
        )

        with open(
            structured_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                elements,
                f,
                indent=2,
                ensure_ascii=False
            )

        # COUNT ELEMENTS
        counts = {

            "pages": page_count,

            "text": sum(
                1 for e in elements
                if e.get("element_type") == "text"
            ),

            "tables": sum(
                1 for e in elements
                if e.get("element_type") == "table"
            ),

            "formulas": sum(
                1 for e in elements
                if e.get("element_type") == "formula"
            ),

            "total": len(elements),

            "layout": layout

        }

        # STEP 3: BUILD SEARCH INDEX
        indexed_count, vocabulary_size = build_faiss_index(
            elements
        )

        faiss.write_index(
            faiss_index,
            os.path.join(output_dir, "rag_index.faiss")
        )

        with open(
            os.path.join(output_dir, "rag_metadata.json"),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        document_elements = elements

        current_filename = safe_name

        document_counts = counts

        message = (
            f"Document processed successfully! "
            f"{indexed_count} elements indexed. "
            f"Vocabulary size: {vocabulary_size}."
        )

        return render_template(

            "index.html",

            message=message,

            filename=current_filename,

            counts=document_counts,

            results=None,

            question=""

        )

    except Exception as e:

        return render_template(

            "index.html",

            message=f"Processing failed: {str(e)}",

            filename=safe_name,

            counts={},

            results=None,

            question=""

        )


# -------------------------------------------------
# ASK QUESTION
# -------------------------------------------------

@app.route("/query", methods=["POST"])
def query_document():

    question = request.form.get(
        "question",
        ""
    ).strip()

    if not question:

        return render_template(
            "index.html",
            message="Please enter a question.",
            filename=current_filename,
            counts=document_counts,
            results=None,
            question=""
        )

    if faiss_index is None or vectorizer is None:

        return render_template(
            "index.html",
            message="Please upload and process a document first.",
            filename=current_filename,
            counts=document_counts,
            results=None,
            question=question
        )

    # STEP 1: RETRIEVE RELEVANT ELEMENTS
    retrieved_results = search_document(
        question,
        top_k=3
    )

    # STEP 2: EXTRACT ONE PARTICULAR ANSWER
    answer = generate_answer(
        question,
        retrieved_results
    )

    if answer:

        results = [answer]

        message = "Answer found successfully."

    else:

        results = []

        message = (
            "No specific matching answer found in the document. "
            "Try rephrasing your question."
        )

    return render_template(

        "index.html",

        message=message,

        filename=current_filename,

        counts=document_counts,

        results=results,

        question=question

    )


# -------------------------------------------------
# RUN FLASK
# -------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)
