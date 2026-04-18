from sentence_transformers import SentenceTransformer
import joblib
import os

DATA_PATH = "data/user_data.txt"
MODEL_PATH = "model/semantic_model.pkl"

model = SentenceTransformer('all-MiniLM-L6-v2')

inputs = []
outputs = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Skip empty or invalid lines
        if not line or "→" not in line:
            continue

        # Split only once (IMPORTANT FIX)
        parts = line.split("→", 1)

        if len(parts) != 2:
            continue

        inp = parts[0].strip()
        out = parts[1].strip()

        # Skip if either side is empty
        if not inp or not out:
            continue

        inputs.append(inp)
        outputs.append(out)

print(f"Loaded {len(inputs)} valid training samples")

if len(inputs) == 0:
    raise ValueError("❌ No valid data found. Check user_data.txt format.")

# Create embeddings (normalized)
embeddings = model.encode(inputs, normalize_embeddings=True)

os.makedirs("model", exist_ok=True)
joblib.dump((embeddings, outputs), MODEL_PATH)

print("✅ Training completed successfully!")