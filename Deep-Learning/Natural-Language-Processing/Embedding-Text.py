import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import re

# Text → Normalize → Tokenize → Map to Indices → Pad → Embedding → Tensor of Word Vectors

# Original text data
texts = [
    "I love this movie! It's fantastic.",
    "I hated the ending... so bad.",
    "Meh, the plot was okay; acting was great."
]

# Normalize text
def normalizing(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

normalized_text = [normalizing(data) for data in texts]
tokens = [data.split(" ") for data in normalized_text]

# Build vocabulary
clean_tokens = sorted({text for data in tokens for text in data})
vocab = {text: idx + 2 for idx, text in enumerate(clean_tokens)}
vocab['<PAD>'] = 0
vocab['<UKN>'] = 1

# Convert tokens to sequences of indices
def text_to_sequence(tokens, vocab):
    sequences = []
    for sentence in tokens:
        sequence = [vocab.get(word, vocab['<UKN>']) for word in sentence]
        sequences.append(torch.tensor(sequence, dtype=torch.long))
    return sequences

sequences = text_to_sequence(tokens, vocab)

# Pad sequences using PyTorch
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab['<PAD>'])

# Result
print("Vocabulary:", vocab)
print("Padded Tensor:\n", padded_sequences)

# Suppose your vocab size
vocab_size = len(vocab)    # include <PAD> and <UKN>
embedding_dim = 50         # typical dimensions: 50, 100, 300, etc.

# Create the embedding layer
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=vocab['<PAD>'])

# Forward pass: convert padded_sequences to embeddings
embedded = embedding(padded_sequences)

print("Embedded shape:", embedded.shape)