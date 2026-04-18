from sentence_transformers import SentenceTransformer
import joblib, os

model = SentenceTransformer('all-MiniLM-L6-v2')

inputs, outputs = [], []

with open("data/user_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "→" not in line:
            continue
        parts = line.split("→", 1)
        inp, out = parts[0].strip(), parts[1].strip()
        if inp and out:
            inputs.append(inp)
            outputs.append(out)

embeddings = model.encode(inputs, normalize_embeddings=True)

os.makedirs("model", exist_ok=True)
joblib.dump((embeddings, outputs), "model/semantic_model.pkl")

print("Chat model ready")