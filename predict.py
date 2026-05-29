# predict.py
import torch, pickle, json
import torch.nn as nn

class SentimentNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# Charger
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/config.json") as f:
    config = json.load(f)

model = SentimentNet(config["input_dim"])
model.load_state_dict(torch.load("model/model.pt"))
model.eval()

# Test
text = ["This movie was absolutely amazing!"]
vec = vectorizer.transform(text).toarray()
tensor = torch.FloatTensor(vec)
pred = model(tensor).argmax(dim=1).item()
print(f"Sentiment: {config['classes'][pred]}")