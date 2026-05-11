"""
decision_utils.py — Decision prediction engine.

Replaces sklearn/joblib/threading with:
  - DB-backed feedback storage (Postgres or SQLite)
  - Keyword overlap for instant feedback matching (no sklearn needed)
  - Google Gemini API for AI-powered decision reasoning
"""

import os
import database

# ── Gemini setup ──────────────────────────────────────────────────────────────

GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
_client = None


def _get_client():
    global _client
    if _client is None and GEMINI_KEY:
        from google import genai as google_genai
        _client = google_genai.Client(api_key=GEMINI_KEY)
    return _client


SIMILARITY_THRESHOLD = 0.40   # keyword overlap threshold


# ── Path helpers (kept for compat — not used in Postgres mode) ────────────────

def get_user_decision_model_path(username):
    return os.path.join("model", "users", username, "decision_model.pkl")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db():
    return database.get_conn()


def log_decision(username, context, option_a, option_b, predicted):
    """Save a new decision prediction. Returns the row id."""
    conn = _db()
    c = conn.cursor()
    if database._USE_POSTGRES:
        c.execute(
            "INSERT INTO decision_feedback (username, context, option_a, option_b, predicted) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (username, context, option_a, option_b, predicted)
        )
        row_id = c.fetchone()[0]
    else:
        c.execute(
            "INSERT INTO decision_feedback (username, context, option_a, option_b, predicted) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, context, option_a, option_b, predicted)
        )
        row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id


def record_feedback(username, decision_id, correct_choice, was_wrong, reason=""):
    """Update a decision row with user feedback."""
    conn = _db()
    c = conn.cursor()

    c.execute(
        database._pg(
            "SELECT context, option_a, option_b "
            "FROM decision_feedback WHERE id = ? AND username = ?"
        ),
        (decision_id, username)
    )
    row = c.fetchone()

    c.execute(
        database._pg(
            "UPDATE decision_feedback "
            "SET correct = ?, was_wrong = ?, reason = ? "
            "WHERE id = ? AND username = ?"
        ),
        (correct_choice, 1 if was_wrong else 0, reason, decision_id, username)
    )
    conn.commit()
    conn.close()

    # Feed confirmed decision into chat model as natural-language Q&A pairs
    if row:
        ctx, opt_a, opt_b = row
        _feed_decision_to_chat(username, ctx, opt_a, opt_b, correct_choice, reason)


def _feed_decision_to_chat(username, context, opt_a, opt_b, correct, reason=""):
    """Convert a confirmed decision into chat Q&A pairs and store them."""
    answer_base = f"Based on your past choices, you prefer {correct} when it comes to {context}."
    answer_with_reason = (
        f"{answer_base} You mentioned: '{reason}'" if reason else answer_base
    )

    pairs = [
        (context,                                              answer_base),
        (f"should I {context}",                               answer_with_reason),
        (f"what do I prefer {context}",                       answer_base),
        (f"{opt_a} or {opt_b}",                              answer_base),
        (f"what would I choose between {opt_a} and {opt_b}", answer_base),
        (f"which do I like more {opt_a} or {opt_b}",         answer_base),
    ]

    try:
        from chat_utils import add_to_user_data
        for q, a in pairs:
            add_to_user_data(username, q, a)
    except Exception:
        pass


def get_feedback_history(username, limit=20):
    conn = _db()
    c = conn.cursor()
    c.execute(
        database._pg(
            "SELECT id, context, option_a, option_b, predicted, correct, was_wrong, reason, timestamp "
            "FROM decision_feedback "
            "WHERE username = ? AND correct IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT ?"
        ),
        (username, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "context": r[1], "option_a": r[2], "option_b": r[3],
            "predicted": r[4], "correct": r[5], "was_wrong": bool(r[6]),
            "reason": r[7], "timestamp": str(r[8])
        }
        for r in rows
    ]


def get_feedback_stats(username):
    conn = _db()
    c = conn.cursor()
    c.execute(
        database._pg(
            "SELECT COUNT(*), SUM(CASE WHEN was_wrong = 0 THEN 1 ELSE 0 END) "
            "FROM decision_feedback WHERE username = ? AND correct IS NOT NULL"
        ),
        (username,)
    )
    row = c.fetchone()
    conn.close()
    total   = row[0] or 0
    correct = int(row[1] or 0)
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


def get_confirmed_training_data(username):
    conn = _db()
    c = conn.cursor()
    c.execute(
        database._pg(
            "SELECT context, option_a, option_b, correct "
            "FROM decision_feedback WHERE username = ? AND correct IS NOT NULL "
            "ORDER BY timestamp DESC"
        ),
        (username,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ── Similarity matching (keyword overlap — no sklearn) ────────────────────────

def find_matching_feedback(username, context, option_a, option_b):
    """
    Find a confirmed past decision semantically similar to the new query.
    Uses keyword overlap instead of TF-IDF (no sklearn on Vercel).
    """
    rows = get_confirmed_training_data(username)
    if not rows:
        return None, 0.0

    query   = f"{context} {option_a} {option_b}".lower()
    q_words = set(query.split())
    best_match = None
    best_score = 0.0

    for ctx, opt_a, opt_b, correct in rows:
        past    = f"{ctx} {opt_a} {opt_b}".lower()
        p_words = set(past.split())
        overlap = len(q_words & p_words)
        score   = overlap / max(len(q_words | p_words), 1)   # Jaccard similarity
        if score > best_score:
            best_score = score
            best_match = correct

    if best_score >= SIMILARITY_THRESHOLD:
        return best_match, best_score
    return None, 0.0


# ── Training (lightweight — no model file needed) ────────────────────────────

def train_user_decision_model(username):
    rows = get_confirmed_training_data(username)
    if not rows:
        return False, "No confirmed decisions yet. Make a decision and rate it first."
    return True, f"Decision model updated from {len(rows)} decision{'s' if len(rows) > 1 else ''}."


def retrain_if_ready(username):
    """Sync retrain — background threads not reliable in serverless."""
    train_user_decision_model(username)


# ── Reason builder ───────────────────────────────────────────────────────────

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


# ── Prediction ───────────────────────────────────────────────────────────────

def predict(context, option_a, option_b, username=None):
    """
    Predict the best decision. Priority:
      1. Instant feedback match   — confirmed past decision
      2. Gemini reasoning         — AI analyses user history + context
      3. Default fallback         — returns option_a with 60% confidence
    """
    # ── 1. Past feedback override ─────────────────────────────────────────────
    if username:
        match, score = find_matching_feedback(username, context, option_a, option_b)
        if match:
            confidence = round(min(score * 140, 99.0), 1)
            reason     = _reason_from_context(context, match, source="feedback")
            return match, confidence, reason

    # ── 2. Gemini reasoning ───────────────────────────────────────────────────
    client = _get_client()
    if client:
        history_hint = ""
        if username:
            rows = get_confirmed_training_data(username)
            if rows:
                snippets = "; ".join(
                    f"chose '{r[3]}' over '{r[2]}' for '{r[0]}'"
                    for r in rows[-5:]
                )
                history_hint = f"\nUser's past decisions: {snippets}."

        prompt = (
            f"You are a digital twin decision engine. Based on the user's profile:{history_hint}\n"
            f"Context: {context}\n"
            f"Option A: {option_a}\n"
            f"Option B: {option_b}\n\n"
            f"Which option would this user most likely choose? "
            f"Reply with ONLY the option text (exact match of A or B), nothing else."
        )
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            raw      = response.text.strip()
            decision = option_b if option_b.lower() in raw.lower() else option_a
            reason   = _reason_from_context(context, decision)
            return decision, 78.0, reason
        except Exception:
            pass

    # ── 3. Default fallback ───────────────────────────────────────────────────
    return option_a, 60.0, "Based on your usual preferences."