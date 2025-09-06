import spacy
from sklearn.feature_extraction.text import CountVectorizer
from typing import List

# Step 1: Define important sentiment-related words
def load_critical_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        words = set(line.strip() for line in f if line.strip())  # remove empty lines
    return words

# Load from file
critical_sentiment_words = load_critical_words('Custom-Words.txt')

# Step 2: Load spaCy and prepare custom stopwords
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words.copy()

# Option A: Remove critical sentiment words from stopwords using set difference
custom_stopwords_A = spacy_stopwords - critical_sentiment_words

# Option B: Remove critical sentiment words from stopwords using discard in a loop
custom_stopwords_B = spacy_stopwords.copy()
for word in critical_sentiment_words:
    custom_stopwords_B.discard(word)

# -------------------------------------------
# Option 1: CountVectorizer with stopwords parameter and bigram support

vectorizer_option_1 = CountVectorizer(
    stop_words=custom_stopwords_A,  # or custom_stopwords_B - they are equivalent here
    ngram_range=(1, 2),             # unigrams and bigrams
    lowercase=True
)

# -------------------------------------------
# Option 2: Custom spaCy tokenizer with CountVectorizer, no stopwords param

def spacy_tokenizer(text: str) -> List[str]:
    """
    Tokenize, lemmatize, remove stopwords (except critical sentiment words),
    punctuation, numbers, non-alpha tokens.
    """
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if (
            token.is_alpha and
            not token.is_stop and
            token.text.lower() not in custom_stopwords_A and
            not token.like_num and
            not token.is_punct and
            not token.is_space
        )
    ]
    return tokens

vectorizer_option_2 = CountVectorizer(
    tokenizer=spacy_tokenizer,
    stop_words=None,        # Already handled in tokenizer
    ngram_range=(1, 2),
    lowercase=True
)

# -------------------------------------------
# Example usage to test both vectorizers

if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this amazing movie! It's a masterpiece.",
        "I don't like this boring and terrible film at all.",
        "It's not a bad movie, but not very good either."
    ]

    print("=== Using vectorizer_option_1 (stopwords param) ===")
    X1 = vectorizer_option_1.fit_transform(sample_texts)
    print("Features:", vectorizer_option_1.get_feature_names_out())
    print("Shape:", X1.shape, "\n")

    print("=== Using vectorizer_option_2 (custom tokenizer) ===")
    X2 = vectorizer_option_2.fit_transform(sample_texts)
    print("Features:", vectorizer_option_2.get_feature_names_out())
    print("Shape:", X2.shape)
