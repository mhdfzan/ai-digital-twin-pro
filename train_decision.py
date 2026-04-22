import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH  = "data/decisions.txt"
MODEL_PATH = "model/decision_model.pkl"

contexts = []
labels   = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) != 4:
            continue
        context, opt_a, opt_b, chosen = parts
        text = f"{context} [A] {opt_a} [B] {opt_b}"
        contexts.append(text)
        labels.append(0 if chosen == opt_a else 1)

if len(contexts) == 0:
    raise ValueError("❌ No valid data in decisions.txt. Format: context | optA | optB | chosen")

print(f"Loaded {len(contexts)} decision samples")

vec = TfidfVectorizer()
X   = vec.fit_transform(contexts)

model = LogisticRegression(max_iter=500)
model.fit(X, labels)

os.makedirs("model", exist_ok=True)
joblib.dump((vec, model), MODEL_PATH)

print("✅ Decision model trained and saved!")