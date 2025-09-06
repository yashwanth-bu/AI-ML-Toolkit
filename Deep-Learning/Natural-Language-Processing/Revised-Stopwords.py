from revised_stopwords import get_revised_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Get the stopwords list (revised stopwords from your custom source)
stopwords_list = list(get_revised_stopwords())  # Convert set to list

# Sample text data for the vectorizer
texts = [
    "The company reported a growth in revenue this quarter.",
    "Investors are optimistic about the stock market performance.",
    "The debt level of the company raised concerns among investors."
]

# Initial TfidfVectorizer with revised stopwords
tdf_vectorizer = TfidfVectorizer(
    stop_words=stopwords_list,  # Pass the list of stopwords
    ngram_range=(1, 2),         # Use unigrams and bigrams
    lowercase=True              # Convert text to lowercase
)

# Fit and transform the text data
tdf_texts = tdf_vectorizer.fit_transform(texts)

# Output: Show the feature names (terms included after applying stopwords)
print("Feature Names with Revised Stopwords:")
print(tdf_vectorizer.get_feature_names_out())

# Output: Show the TF-IDF matrix
print("\nTF-IDF Matrix (as array):")
print(tdf_texts.toarray())


# Custom stopwords list for the stock market context
custom_words = [
    'company', 'stock', 'investors', 'market', 'growth', 'performance', 'revenue', 'quarter', 'price', 'share', 'shares', 'value', 'capital', 'business', 'debt', 'financial', 'thing', 'stuff', 'way', 'time', 'place', 'matter', 'fact', 'point', 'result', 'condition', 'to', 'from', 'about', 'as'
]

# Combine revised stopwords with custom words and sort the list
custom_stopword_list = sorted(set(stopwords_list + custom_words))

# Create a new TfidfVectorizer using the custom stopword list
custom_tdf_vectorizer = TfidfVectorizer(
    stop_words=custom_stopword_list,  # Use custom stopwords list
    ngram_range=(1, 2),              # Use unigrams and bigrams
    lowercase=True                   # Convert text to lowercase
)

# Fit and transform the text data using the custom stopword list
custom_tdf_texts = custom_tdf_vectorizer.fit_transform(texts)

# Output: Show the feature names (terms included after applying custom stopwords)
print("\nFeature Names with Custom Stopwords:")
print(custom_tdf_vectorizer.get_feature_names_out())

# Output: Show the TF-IDF matrix for custom stopwords
print("\nTF-IDF Matrix with Custom Stopwords (as array):")
print(custom_tdf_texts.toarray())
