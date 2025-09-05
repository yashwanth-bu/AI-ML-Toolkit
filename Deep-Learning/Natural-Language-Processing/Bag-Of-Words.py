import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
texts = [
    "I love this movie! It's fantastic.",
    "I hated the ending... so bad.",
    "Meh, the plot was okay; acting was great."
]

# ----------- Manual Bag of Words -----------
print("=== Manual Bag of Words Implementation ===")

# Normalize text
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

# Create bag of words vectors
manual_bow_vectors = []
for sentence in tokens:
    word_counts = Counter(sentence)
    bow_vector = [word_counts.get(word, 0) for word in manual_vocab]
    manual_bow_vectors.append(bow_vector)

print("Vocabulary:", manual_vocab)
print("Bag of Words vectors:")
for vector in manual_bow_vectors:
    print(vector)

# ----------- Scikit-learn CountVectorizer -----------
print("\n=== Scikit-learn CountVectorizer Implementation ===")

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform texts
bow_matrix = vectorizer.fit_transform(texts)

# Get vocabulary
sklearn_vocab = vectorizer.get_feature_names_out()

# Convert to dense array
sklearn_bow_vectors = bow_matrix.toarray()

print("Vocabulary:", list(sklearn_vocab))
print("Bag of Words vectors:")
for vector in sklearn_bow_vectors:
    print(vector.tolist())
