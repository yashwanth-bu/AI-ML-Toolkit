import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
texts = [
    "I love this movie! It's fantastic.",
    "I hated the ending... so bad.",
    "Meh, the plot was okay; acting was great."
]

# ----------- Manual TF-IDF Implementation -----------
print("=== Manual TF-IDF Implementation ===")

# Text normalization
def normalizing(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Normalize and tokenize
normalized_texts = [normalizing(text) for text in texts]
tokens = [text.split(" ") for text in normalized_texts]

# Build vocabulary
manual_vocab = sorted(set(word for sentence in tokens for word in sentence))

# Compute Document Frequency (DF)
df = {}
N = len(tokens)  # total number of documents

for word in manual_vocab:
    df[word] = sum(1 for doc in tokens if word in doc)

# Compute TF-IDF vectors
manual_tfidf_vectors = []

for doc in tokens:
    tf = Counter(doc)
    doc_len = len(doc)
    tfidf_vector = []
    for word in manual_vocab:
        tf_word = tf[word] / doc_len
        idf_word = math.log((N + 1) / (df[word] + 1)) + 1  # Smoothed IDF
        tfidf_vector.append(tf_word * idf_word)
    manual_tfidf_vectors.append(tfidf_vector)

print("Vocabulary:", manual_vocab)
print("TF-IDF vectors:")
for vec in manual_tfidf_vectors:
    print([round(x, 3) for x in vec])

# ----------- Scikit-learn TF-IDF Implementation -----------
print("\n=== Scikit-learn TfidfVectorizer Implementation ===")

# Initialize vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(texts)

# Get vocabulary
sklearn_vocab = vectorizer.get_feature_names_out()

# Convert matrix to array
sklearn_tfidf_vectors = tfidf_matrix.toarray()

print("Vocabulary:", list(sklearn_vocab))
print("TF-IDF vectors:")
for vec in sklearn_tfidf_vectors:
    print([round(x, 3) for x in vec])
