"""
train_chat.py

Usage:
    python train_chat.py --user faizan    # train for specific user
    python train_chat.py                   # train global fallback model
"""
import sys, os, joblib
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def train_for_user(username):
    data_path  = os.path.join("data", "users", username, "user_data.txt")
    model_path = os.path.join("model", "users", username, "semantic_model.pkl")

    if not os.path.exists(data_path):
        print(f"❌ No data at {data_path}"); return

    inputs, outputs = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "→" not in line: continue
            parts = line.split("→", 1)
            if len(parts) != 2: continue
            inp, out = parts[0].strip(), parts[1].strip()
            if inp and out:
                inputs.append(inp); outputs.append(out)

    if not inputs:
        print("❌ No valid pairs found."); return

    print(f"Training '{username}' on {len(inputs)} pairs...")
    emb = MODEL.encode(inputs, normalize_embeddings=True, show_progress_bar=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((emb, outputs), model_path)
    print(f"✅ Saved to {model_path}")


def train_global():
    data_path  = "data/user_data.txt"
    model_path = "model/semantic_model.pkl"

    if not os.path.exists(data_path):
        print(f"❌ No data at {data_path}"); return

    inputs, outputs = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "→" not in line: continue
            parts = line.split("→", 1)
            if len(parts) != 2: continue
            inp, out = parts[0].strip(), parts[1].strip()
            if inp and out:
                inputs.append(inp); outputs.append(out)

    if not inputs:
        print("❌ No valid pairs found."); return

    print(f"Training global model on {len(inputs)} pairs...")
    emb = MODEL.encode(inputs, normalize_embeddings=True, show_progress_bar=True)
    os.makedirs("model", exist_ok=True)
    joblib.dump((emb, outputs), model_path)
    print(f"✅ Saved to {model_path}")


if __name__ == "__main__":
    if "--user" in sys.argv:
        idx = sys.argv.index("--user")
        train_for_user(sys.argv[idx + 1])
    else:
        train_global()