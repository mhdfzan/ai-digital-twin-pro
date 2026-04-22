import joblib
import random
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

_encoder     = None
_user_models = {}   # { username: (embeddings, outputs) }


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def get_user_model_path(username):
    return os.path.join("model", "users", username, "semantic_model.pkl")


def get_user_data_path(username):
    return os.path.join("data", "users", username, "user_data.txt")


def get_global_model_path():
    return os.path.join("model", "semantic_model.pkl")


def load_user_model(username):
    if username in _user_models:
        return _user_models[username]
    path = get_user_model_path(username)
    if os.path.exists(path):
        emb, out = joblib.load(path)
        _user_models[username] = (emb, out)
        return emb, out
    # fallback
    gp = get_global_model_path()
    if os.path.exists(gp):
        return joblib.load(gp)
    return None, None


def invalidate_user_model(username):
    if username in _user_models:
        del _user_models[username]


def train_user_model(username):
    data_path = get_user_data_path(username)
    if not os.path.exists(data_path):
        return False, "No training data found."

    inputs, outputs = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "→" not in line:
                continue
            parts = line.split("→", 1)
            if len(parts) != 2:
                continue
            inp, out = parts[0].strip(), parts[1].strip()
            if inp and out:
                inputs.append(inp)
                outputs.append(out)

    if len(inputs) < 3:
        return False, f"Need at least 3 training pairs. You have {len(inputs)}."

    encoder    = _get_encoder()
    embeddings = encoder.encode(inputs, normalize_embeddings=True)

    model_dir  = os.path.dirname(get_user_model_path(username))
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump((embeddings, outputs), get_user_model_path(username))

    invalidate_user_model(username)
    return True, f"Twin trained on {len(inputs)} conversation pairs."


def ensure_user_data_file(username, seed_data=False):
    data_path = get_user_data_path(username)
    if os.path.exists(data_path):
        return
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if seed_data:
        # Minimal fallback — onboarding will enrich this
        starter = "hi → hey!\nhello → hey there!\nhow are you → I'm good!\n"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(starter)
    else:
        open(data_path, "w", encoding="utf-8").close()


def write_pairs_to_user_data(username, pairs):
    """Write a list of (input, output) tuples to user's dataset."""
    data_path = get_user_data_path(username)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "a", encoding="utf-8") as f:
        for inp, out in pairs:
            f.write(f"{inp} → {out}\n")


def add_to_user_data(username, user_input, bot_reply):
    data_path = get_user_data_path(username)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "a", encoding="utf-8") as f:
        f.write(f"{user_input.strip()} → {bot_reply.strip()}\n")


def count_user_data(username):
    data_path = get_user_data_path(username)
    if not os.path.exists(data_path):
        return 0
    count = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if "→" in line.strip():
                count += 1
    return count


def chat_reply(user_input, username=None):
    encoder = _get_encoder()

    if username:
        embeddings, outputs = load_user_model(username)
    else:
        embeddings, outputs = None, None
        gp = get_global_model_path()
        if os.path.exists(gp):
            embeddings, outputs = joblib.load(gp)

    if embeddings is None or outputs is None:
        return "My twin hasn't been trained yet!", 0.0

    if not user_input.strip():
        return "Say something! 😄", 0.0

    query  = encoder.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    sims   = util.cos_sim(query, embeddings)[0]
    scores = sims.cpu().numpy()

    top_idx = scores.argsort()[-3:][::-1]
    chosen  = random.choice(top_idx)

    reply      = outputs[chosen]
    confidence = float(scores[chosen])

    if confidence < 0.35:
        return "I'm not sure what to say 😅", confidence

    return reply, confidence