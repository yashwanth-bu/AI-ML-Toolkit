import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import spacy
import pandas as pd
from typing import List
from revised_stopwords import get_revised_stopwords 

# -------------------- 1. Load Critical Words --------------------
def load_critical_words(filepath: str) -> set:
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

# -------------------- 2. Setup NLP + Stopwords --------------------
nlp = spacy.load("en_core_web_sm")
stopwords_list = sorted((nlp.Defaults.stop_words | get_revised_stopwords()) - load_critical_words('Small-Sentiment-Words.txt'))

def spacy_tokenizer(text: str) -> List[str]:
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if (
            token.is_alpha and
            token.text.lower() not in stopwords_list and
            not token.like_num and
            not token.is_punct and
            not token.is_space
        )
    ]

# -------------------- 3. Load Data --------------------
df = pd.read_csv('movie-reviews-200.csv')  # Columns: 'text', 'label'
texts = df['text']
labels = df['label']

# -------------------- 4. Tokenization --------------------
tokens = [spacy_tokenizer(sentence) for sentence in texts]
clean_tokens = sorted({word for sentence in tokens for word in sentence})

# -------------------- 5. Vocabulary --------------------
vocab = {word: idx + 2 for idx, word in enumerate(clean_tokens)}
vocab['<PAD>'] = 0
vocab['<UKN>'] = 1

def text_to_sequence(tokens, vocab):
    return [[vocab.get(word, vocab['<UKN>']) for word in sentence] for sentence in tokens]

sequences = [torch.tensor(seq, dtype=torch.long) for seq in text_to_sequence(tokens, vocab)]
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab['<PAD>'])

# -------------------- 6. Labels --------------------
tensor_y = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)

# -------------------- 7. Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, tensor_y, test_size=0.2, stratify=labels
)

# -------------------- 8. Model Definition --------------------
class SimpleTextNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.model = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.embedding(x)         # [batch_size, seq_len, emb_dim]
        x = x.mean(dim=1)             # Mean pooling → [batch_size, emb_dim]
        return self.model(x)          # [batch_size, 1]

# -------------------- 9. Training Setup --------------------
model = SimpleTextNN(vocab_size=len(vocab), emb_dim=50, padding_idx=vocab['<PAD>'])
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

from torch.utils.data import Dataset, DataLoader

# -------------------- 10. Dataset And DataLoader --------------------
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = SimpleDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

# -------------------- 11. Training Loop --------------------
epochs_loss = []
validation_loss = []
number_epochs = 10

for epoch in range(number_epochs):
    epoch_losses = []

    model.train()
    for input_feature, output_target in train_loader:
        optimizer.zero_grad()
        output = model(input_feature)
        loss = loss_function(output, output_target)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item()) 

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    epochs_loss.append(avg_epoch_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        val_loss = loss_function(preds, y_test)
        validation_loss.append(val_loss.item()) 

    print(f"Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss.item():.4f}")

# -------------------- 12. Evaluation --------------------
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_class = (preds > 0.5).float()
    print(f"\nTest Accuracy: {accuracy_score(y_test, preds_class)}")
    print(f"Confusion Matrix : \n{confusion_matrix(y_test, preds_class)}")
    print(f"Classification Report : \n {classification_report(y_test, preds_class)}")

# -------------------- 13. Visual Plot --------------------
import matplotlib.pyplot as plt

plt.plot(epochs_loss, label="Train Loss")
plt.plot(validation_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()