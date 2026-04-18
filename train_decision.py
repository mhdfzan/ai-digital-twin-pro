import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

contexts, labels = [], []

with open("data/decisions.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "|" not in line:
            continue
        c, a, b, choice = [x.strip() for x in line.split("|")]

        text = f"{c} [A] {a} [B] {b}"
        contexts.append(text)
        labels.append(0 if choice == a else 1)

vec = TfidfVectorizer()
X = vec.fit_transform(contexts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump((vec, model), "model/decision_model.pkl")

print("Decision model ready")