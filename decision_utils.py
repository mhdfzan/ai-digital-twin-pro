import joblib
import os
import sqlite3
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# ─── Paths ────────────────────────────────────────────────────────────────────

GLOBAL_MODEL_PATH = "model/decision_model.pkl"
_global_vec   = None
_global_model = None
_user_models  = {}   # { username: (vec, model) }
_lock         = threading.Lock()

# Similarity threshold: 0.55+ = "close enough" to use past feedback directly
SIMILARITY_THRESHOLD = 0.55


def get_user_decision_model_path(username):
    return os.path.join("model", "users", username, "decision_model.pkl")


# ─── Load Helpers ─────────────────────────────────────────────────────────────

def _load_global():
    global _global_vec, _global_model
    if _global_vec is None and os.path.exists(GLOBAL_MODEL_PATH):
        _global_vec, _global_model = joblib.load(GLOBAL_MODEL_PATH)


def _load_user(username):
    """Load a user's personal decision model into cache."""
    if username in _user_models:
        return _user_models[username]
    path = get_user_decision_model_path(username)
    if os.path.exists(path):
        vec, model = joblib.load(path)
        _user_models[username] = (vec, model)
        return vec, model
    return None, None


def _invalidate_user(username):
    with _lock:
        if username in _user_models:
            del _user_models[username]


# ─── Feedback DB Helpers ───────────────────────────────────────────────────────

def _db():
    return sqlite3.connect("app.db")


def log_decision(username, context, option_a, option_b, predicted):
    """Save a new decision prediction to the feedback table. Returns the row id."""
    conn = _db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO decision_feedback (username, context, option_a, option_b, predicted)
        VALUES (?, ?, ?, ?, ?)
    """, (username, context, option_a, option_b, predicted))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def record_feedback(username, decision_id, correct_choice, was_wrong, reason=""):
    """Update a decision row with user feedback, then feed it to the chat model."""
    conn = _db()
    c = conn.cursor()

    # Fetch the original context before updating so we can build chat pairs
    c.execute("""
        SELECT context, option_a, option_b
        FROM decision_feedback WHERE id = ? AND username = ?
    """, (decision_id, username))
    row = c.fetchone()

    c.execute("""
        UPDATE decision_feedback
        SET correct = ?, was_wrong = ?, reason = ?
        WHERE id = ? AND username = ?
    """, (correct_choice, 1 if was_wrong else 0, reason, decision_id, username))
    conn.commit()
    conn.close()

    # ── Direction 2: Decision → Chat bridge ────────────────────────────
    # Feed the confirmed decision into the chatbot as a training pair so that
    # the chatbot can answer questions about this topic in natural language.
    if row:
        ctx, opt_a, opt_b = row
        _feed_decision_to_chat(username, ctx, opt_a, opt_b, correct_choice, reason)


def _feed_decision_to_chat(username, context, opt_a, opt_b, correct, reason=""):
    """
    Convert a confirmed decision into natural-language chat Q&A pairs and
    inject them into the user's chat model immediately (background thread).
    Multiple phrasings are added so the chatbot can answer in many ways.
    Uses a lazy import of chat_utils to avoid circular imports.
    """
    answer_base = f"Based on your past choices, you prefer {correct} when it comes to {context}."
    if reason:
        answer_with_reason = f"{answer_base} You mentioned: '{reason}'"
    else:
        answer_with_reason = answer_base

    # Multiple question phrasings for better coverage
    pairs = [
        (context,                                           answer_base),
        (f"should I {context}",                            answer_with_reason),
        (f"what do I prefer {context}",                    answer_base),
        (f"{opt_a} or {opt_b}",                           answer_base),
        (f"what would I choose between {opt_a} and {opt_b}", answer_base),
        (f"which do I like more {opt_a} or {opt_b}",      answer_base),
    ]

    try:
        # Lazy import avoids circular dependency
        from chat_utils import learn_from_chat_background, add_to_user_data
        for q, a in pairs:
            add_to_user_data(username, q, a)          # persist to dataset file
            learn_from_chat_background(username, q, a)  # update model in background
    except Exception:
        pass


def get_feedback_history(username, limit=20):
    """Return recent decisions with ratings for the history panel."""
    conn = _db()
    c = conn.cursor()
    c.execute("""
        SELECT id, context, option_a, option_b, predicted, correct, was_wrong, reason, timestamp
        FROM decision_feedback
        WHERE username = ? AND correct IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
    """, (username, limit))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "context": r[1], "option_a": r[2], "option_b": r[3],
            "predicted": r[4], "correct": r[5], "was_wrong": bool(r[6]),
            "reason": r[7], "timestamp": r[8]
        }
        for r in rows
    ]


def get_feedback_stats(username):
    """Return accuracy stats for the user's decision history."""
    conn = _db()
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*), SUM(CASE WHEN was_wrong = 0 THEN 1 ELSE 0 END)
        FROM decision_feedback
        WHERE username = ? AND correct IS NOT NULL
    """, (username,))
    row = c.fetchone()
    conn.close()
    total   = row[0] or 0
    correct = row[1] or 0
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


def get_confirmed_training_data(username):
    """Fetch all confirmed feedback rows to use for training."""
    conn = _db()
    c = conn.cursor()
    c.execute("""
        SELECT context, option_a, option_b, correct
        FROM decision_feedback
        WHERE username = ? AND correct IS NOT NULL
        ORDER BY timestamp DESC
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows


# ─── Instant Feedback Override (works from feedback #1) ───────────────────────

def find_matching_feedback(username, context, option_a, option_b):
    """
    Semantic similarity lookup over all confirmed past feedback.
    If a past decision is similar enough, return its confirmed answer directly.
    This fires even after the very first feedback — no minimum threshold.
    Returns (correct_choice, similarity_score) or (None, 0) if no match.
    """
    rows = get_confirmed_training_data(username)
    if not rows:
        return None, 0.0

    query = f"{context} {option_a} {option_b}".lower().strip()
    past_texts = [
        f"{r[0]} {r[1]} {r[2]}".lower().strip()
        for r in rows
    ]

    # Build TF-IDF vectors just for comparison
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        all_texts = [query] + past_texts
        tfidf_matrix = vec.fit_transform(all_texts)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            # Most recent matching feedback wins
            matched_row = rows[best_idx]
            # ctx, opt_a, opt_b, correct = matched_row
            return matched_row[3], best_score   # correct choice
    except Exception:
        pass

    return None, 0.0


# ─── Training ─────────────────────────────────────────────────────────────────

def train_user_decision_model(username):
    """
    Retrain the per-user decision model from all confirmed feedback.
    Works with as little as 1 sample (uses class_weight to handle imbalance).
    Returns (success: bool, message: str).
    """
    rows = get_confirmed_training_data(username)
    if len(rows) < 1:
        return False, "No confirmed decisions yet. Make a decision and rate it first."

    texts, labels = [], []
    for ctx, opt_a, opt_b, correct in rows:
        text = f"{ctx} [A] {opt_a} [B] {opt_b}"
        texts.append(text)
        labels.append(0 if correct.strip().lower() == opt_a.strip().lower() else 1)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X   = vec.fit_transform(texts)

    # Use balanced class_weight so even a single sample of each class trains well
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X, labels)

    model_dir = os.path.dirname(get_user_decision_model_path(username))
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump((vec, model), get_user_decision_model_path(username))

    _invalidate_user(username)
    return True, f"Model updated from {len(rows)} decision{'s' if len(rows) > 1 else ''}."


def retrain_if_ready(username):
    """
    Trigger retrain in background thread after EVERY feedback — no minimum wait.
    The semantic override already handles first-try accuracy, but the ML model
    also improves continuously from feedback #1.
    """
    t = threading.Thread(target=train_user_decision_model, args=(username,), daemon=True)
    t.start()


# ─── Reason Builder ───────────────────────────────────────────────────────────

def _reason_from_context(context, decision, source="model"):
    if source == "feedback":
        return "You told me this before — I remembered your preference. ✨"
    ctx = context.lower()
    if any(w in ctx for w in ["money", "buy", "spend", "save", "cost"]):
        return "You tend to be mindful about spending."
    elif any(w in ctx for w in ["study", "work", "assignment", "productive"]):
        return "You usually prioritize productivity."
    elif any(w in ctx for w in ["home", "stay", "rest", "relax", "sleep"]):
        return "You generally prefer staying comfortable."
    elif any(w in ctx for w in ["gym", "exercise", "workout", "health"]):
        return "You value keeping healthy habits."
    elif any(w in ctx for w in ["eat", "food", "hungry", "meal"]):
        return "Based on your usual food preferences."
    else:
        return "Based on your past decision patterns."


# ─── Prediction (priority: feedback override → user model → global → default) ──

def predict(context, option_a, option_b, username=None):
    """
    Predict the best decision. Priority:
      1. Instant feedback override  — similar past decision the user confirmed
      2. User's trained ML model    — trained on all their confirmed feedback
      3. Global shared model        — generic trained model
      4. Default fallback           — returns option_a with 60% confidence
    Returns (decision, confidence, reason).
    """

    # ── 1. Instant feedback override (works from the very first feedback) ──────
    if username:
        match, score = find_matching_feedback(username, context, option_a, option_b)
        if match:
            confidence = round(min(score * 110, 99.0), 1)  # scale sim → confidence
            reason = _reason_from_context(context, match, source="feedback")
            return match, confidence, reason

    # ── 2. User's personal trained ML model ───────────────────────────────────
    if username:
        vec, model = _load_user(username)
        if vec is not None and model is not None:
            text  = f"{context} [A] {option_a} [B] {option_b}"
            X     = vec.transform([text])
            probs = model.predict_proba(X)[0]
            idx   = int(probs.argmax())
            decision   = option_a if idx == 0 else option_b
            confidence = round(float(probs.max()) * 100, 1)
            reason     = _reason_from_context(context, decision, source="model")
            return decision, confidence, reason

    # ── 3. Global shared model ────────────────────────────────────────────────
    _load_global()
    if _global_vec is not None and _global_model is not None:
        text  = f"{context} [A] {option_a} [B] {option_b}"
        X     = _global_vec.transform([text])
        probs = _global_model.predict_proba(X)[0]
        idx   = int(probs.argmax())
        decision   = option_a if idx == 0 else option_b
        confidence = round(float(probs.max()) * 100, 1)
        reason     = _reason_from_context(context, decision, source="model")
        return decision, confidence, reason

    # ── 4. Default fallback ───────────────────────────────────────────────────
    return option_a, 60.0, "Based on your usual preferences."