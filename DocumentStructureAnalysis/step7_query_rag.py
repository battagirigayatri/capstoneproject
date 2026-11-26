import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load extracted document
with open("rag_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Break the large document into small pieces
small_chunks = []

for item in metadata:
    content = item.get("content", "")

    # Split at line breaks
    parts = re.split(r'[\n\r]+', content)

    # Also split long text at sentence boundaries
    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Break very long pieces into sentences
        sentences = re.split(r'(?<=[.!?])\s+', part)

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) > 5:
                small_chunks.append({
                    "content": sentence,
                    "source_page": item.get("source_page"),
                    "element_type": item.get("element_type"),
                    "confidence_score": item.get("confidence_score")
                })

if not small_chunks:
    print("No usable text found.")
    exit()

texts = [x["content"] for x in small_chunks]

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    stop_words=None
)

vectors = vectorizer.fit_transform(texts)

print("======================================")
print("       DOCUMENT RAG QUERY SYSTEM")
print("======================================")
print(f"Searchable chunks: {len(texts)}")
print("Embedding method: TF-IDF")
print()

question = input("Enter your question: ")

query_vector = vectorizer.transform([question])

scores = cosine_similarity(query_vector, vectors)[0]

# Get best matching result
best_index = scores.argmax()

best = small_chunks[best_index]

print()
print("========== RETRIEVED RESULT ==========")
print()
print("Question:")
print(question)

print()
print("Retrieved Content:")
print(best["content"])

print()
print("Source Page:", best["source_page"])
print("Element Type:", best["element_type"])
print("Confidence:", best["confidence_score"])
print("Relevance Score:", round(float(scores[best_index]), 4))

print()
print("======================================")