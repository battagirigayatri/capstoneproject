import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# Load structured document
with open("output_combined_structured.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

# Keep only elements containing actual text
valid_elements = []

for item in elements:
    text = str(item.get("content", "")).strip()
    if text:
        valid_elements.append(item)

if not valid_elements:
    print("ERROR: No usable text found in output_combined_structured.json")
    exit()

texts = [str(item["content"]) for item in valid_elements]

print(f"Loaded {len(texts)} text elements.")

# Create TF-IDF embeddings
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    stop_words=None
)

embeddings = vectorizer.fit_transform(texts).toarray().astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "rag_index.faiss")

# Save metadata
metadata = []

for item in valid_elements:
    metadata.append({
        "content": item.get("content", ""),
        "source_page": item.get("source_page"),
        "element_type": item.get("element_type", "unknown"),
        "confidence_score": item.get("confidence_score")
    })

with open("rag_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Save vocabulary
with open("rag_vectorizer.json", "w", encoding="utf-8") as f:
    json.dump(vectorizer.vocabulary_, f, indent=2)

print()
print("RAG index created successfully!")
print(f"Total indexed elements: {len(valid_elements)}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print("Created: rag_index.faiss")
print("Created: rag_metadata.json")
print("Embedding method: TF-IDF")