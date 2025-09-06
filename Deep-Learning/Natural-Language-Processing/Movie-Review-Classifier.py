import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import spacy
from typing import List
from revised_stopwords import get_revised_stopwords

# Load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = list(nlp.Defaults.stop_words)

def load_critical_words(filepath: str) -> set:
    """
    Loads critical sentiment-related words from a file.
    
    Args:
        filepath (str): Path to the file containing the critical words.
        
    Returns:
        set: A set of critical words.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        words = set(line.strip() for line in f if line.strip())  # Remove empty lines
    return words

def spacy_tokenizer(text: str) -> List[str]:
    """
    Tokenizes the input text using spaCy. Removes stopwords, lemmatizes words,
    and filters out non-alphabetic tokens, numbers, punctuation, and spaces.

    Args:
        text (str): The input text to tokenize.

    Returns:
        List[str]: A list of processed tokens.
    """
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()  # Convert to lowercase and lemmatize
        for token in doc
        if (
            token.is_alpha and  # Keep only alphabetic tokens
            not token.is_stop and  # Remove stopwords
            token.text.lower() not in stopwords_list and  # Remove custom stopwords
            not token.like_num and  # Exclude numbers
            not token.is_punct and  # Exclude punctuation
            not token.is_space  # Exclude spaces
        )
    ]
    return tokens

# Load the data
df = pd.read_csv('movie-reviews-200.csv')

# Load additional stopwords
revised_stopwords_list = list(get_revised_stopwords())
critical_sentiment_words = list(load_critical_words('Small-Sentiment-Words.txt'))
stopwords_list = sorted(set(revised_stopwords_list + critical_sentiment_words + spacy_stopwords))

# Prepare data for training and testing
texts = df['text']
labels = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# Vectorize using CountVectorizer
bow_vectorizer = CountVectorizer(
    tokenizer=spacy_tokenizer,
    stop_words=None,  
    ngram_range=(1, 2)  # Use unigrams and bigrams
)

# Transform training and test data
bow_train = bow_vectorizer.fit_transform(X_train)
bow_test = bow_vectorizer.transform(X_test)

# Train the model using Logistic Regression
bow_model = LogisticRegression(class_weight='balanced', max_iter=1000)
bow_model.fit(bow_train, y_train)

# Predict on test data
bow_preds = bow_model.predict(bow_test)

# Evaluate the model
print("BOW Accuracy:", accuracy_score(y_test, bow_preds))
print("BOW Classification Report:\n", classification_report(y_test, bow_preds))

# Vectorize using TfidfVectorizer
tfd_vectorizer = TfidfVectorizer(
    tokenizer=spacy_tokenizer,
    stop_words=None,
    ngram_range=(1, 2)  # Use unigrams and bigrams
)

# Transform training and test data
tfd_train = tfd_vectorizer.fit_transform(X_train)
tfd_test = tfd_vectorizer.transform(X_test)

# Train the model using Logistic Regression
tfd_model = LogisticRegression(class_weight='balanced', max_iter=1000)
tfd_model.fit(tfd_train, y_train)

# Predict on test data
tfd_pred = tfd_model.predict(tfd_test)

# Evaluate the model
print("TDF Accuracy:", accuracy_score(y_test, tfd_pred))
print("TDF Classification Report:\n", classification_report(y_test, tfd_pred))