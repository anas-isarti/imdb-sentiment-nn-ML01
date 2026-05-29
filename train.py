# train.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Charger les données
df = pd.read_csv("data/imdb_balanced_10k.csv")
X = df["review"].values
y = (df["sentiment"] == "positive").astype(int).values

# 2. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X).toarray()

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# 4. Dataset PyTorch
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test)

# 5. Modèle Feedforward
class SentimentNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = SentimentNet(5000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 6. Entraînement
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

# 7. Évaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    acc = (preds == y_test_t).float().mean().item()
print(f"Accuracy: {acc:.4f}")

# 8. Sauvegarde des artifacts
import os
os.makedirs("model", exist_ok=True)

torch.save(model.state_dict(), "model/model.pt")
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/config.json", "w") as f:
    json.dump({"input_dim": 5000, "classes": ["negative", "positive"]}, f)
with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": round(acc, 4)}, f)

print("Artifacts sauvegardés dans model/")