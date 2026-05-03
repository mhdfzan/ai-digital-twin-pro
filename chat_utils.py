import joblib
import random
import os
import threading
import numpy as np
from sentence_transformers import SentenceTransformer, util

_encoder     = None
_user_models = {}   # { username: (embeddings, outputs) }
_learn_lock  = threading.Lock()


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


def learn_from_chat(username, user_input, bot_reply):
    """
    Incrementally update the user's semantic model with ONE new chat pair.
    Only encodes the new message (O(1)) — does NOT re-encode the entire dataset.
    This makes the twin smarter after every single message, instantly.
    """
    with _learn_lock:
        encoder = _get_encoder()

        # Encode only the new input
        new_emb = encoder.encode([user_input.strip()], normalize_embeddings=True)

        path = get_user_model_path(username)
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)

        if os.path.exists(path):
            try:
                existing_embs, existing_outputs = joblib.load(path)
                updated_embs    = np.vstack([existing_embs, new_emb])
                updated_outputs = list(existing_outputs) + [bot_reply.strip()]
            except Exception:
                # Corrupted model — start fresh
                updated_embs    = new_emb
                updated_outputs = [bot_reply.strip()]
        else:
            updated_embs    = new_emb
            updated_outputs = [bot_reply.strip()]

        joblib.dump((updated_embs, updated_outputs), path)

        # Update the in-memory cache immediately so next reply uses new knowledge
        _user_models[username] = (updated_embs, updated_outputs)


def learn_from_chat_background(username, user_input, bot_reply):
    """
    Non-blocking wrapper — runs learn_from_chat in a daemon thread
    so the chat HTTP response is never delayed by model updates.
    """
    t = threading.Thread(
        target=learn_from_chat,
        args=(username, user_input, bot_reply),
        daemon=True
    )
    t.start()


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


# ─── Decision Feedback Bridge ───────────────────────────────────────────────
# Direction: Chat → Decision Feedback
# When a chat message resembles a past decision context, answer from real history.

CHAT_DECISION_SIM_THRESHOLD = 0.38   # lower than decision→decision to be more sensitive


def search_decision_feedback(username, user_input):
    """
    Look for confirmed past decisions that are semantically similar to the
    chat message. If found, return a natural-language answer built from the
    user's real decision history.

    Returns (reply_text: str, confidence: float) or (None, 0.0).
    Uses a lazy import of decision_utils to avoid circular imports.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos

        # Lazy import to avoid circular dependency
        from decision_utils import get_confirmed_training_data
        rows = get_confirmed_training_data(username)
        if not rows:
            return None, 0.0

        query = user_input.lower().strip()
        # Build search corpus from decision context + both options
        past_texts = [f"{r[0]} {r[1]} {r[2]}".lower().strip() for r in rows]

        all_texts = [query] + past_texts
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform(all_texts)
        sims = sk_cos(matrix[0:1], matrix[1:])[0]

        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= CHAT_DECISION_SIM_THRESHOLD:
            ctx, opt_a, opt_b, correct = rows[best_idx]
            confidence = round(min(best_score * 105, 96.0), 1)

            # Build a natural, human-sounding answer referencing the decision
            answer = (
                f"Based on your past choices, you prefer **{correct}** "
                f"when it comes to '{ctx}'. "
                f"You've weighed {opt_a} vs {opt_b} before and went with {correct}. 🎯"
            )
            return answer, confidence

    except Exception:
        pass

    return None, 0.0


def chat_reply(user_input, username=None):
    encoder = _get_encoder()

    # ── Priority 0: Answer from Decision Feedback history ────────────────────
    # If the user's question resembles a past decision they've rated,
    # give them a personalised answer from their real decision history.
    if username and user_input.strip():
        dec_reply, dec_conf = search_decision_feedback(username, user_input)
        if dec_reply:
            return dec_reply, dec_conf

    # ── Priority 1: Personal semantic chat model ───────────────────────────
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