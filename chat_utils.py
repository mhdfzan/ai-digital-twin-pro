"""
chat_utils.py — Gemini-powered chat twin.

Replaces sentence-transformers + local .pkl files with:
  - Google Gemini API  (generates contextual replies as the user's twin)
  - Postgres / SQLite  (stores conversation pairs in user_chat_data table)

Falls back to a simple keyword match when Gemini key is not set (local dev).
"""

import os
import random
import google.generativeai as genai
import database

# ── Gemini setup ─────────────────────────────────────────────────────────────

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
_gemini_model = None


def _get_model():
    global _gemini_model
    if _gemini_model is None and GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        _gemini_model = genai.GenerativeModel("gemini-pro")
    return _gemini_model


# ── Path helpers (kept for backward compat — no longer write files) ───────────

def get_user_model_path(username):
    return os.path.join("model", "users", username, "semantic_model.pkl")


def get_user_data_path(username):
    return os.path.join("data", "users", username, "user_data.txt")


def get_global_model_path():
    return os.path.join("model", "semantic_model.pkl")


# ── Data helpers (DB-backed) ──────────────────────────────────────────────────

def ensure_user_data_file(username, seed_data=False):
    """No-op on Postgres; still creates local file for SQLite dev mode."""
    if not database._USE_POSTGRES:
        data_path = get_user_data_path(username)
        if not os.path.exists(data_path):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            if seed_data:
                starter = "hi → hey!\nhello → hey there!\nhow are you → I'm good!\n"
                with open(data_path, "w", encoding="utf-8") as f:
                    f.write(starter)
            else:
                open(data_path, "w", encoding="utf-8").close()


def add_to_user_data(username, user_input, bot_reply):
    """Persist a chat pair in the database."""
    conn = database.get_conn()
    c = conn.cursor()
    c.execute(
        database._pg("INSERT INTO user_chat_data (username, input, output) VALUES (?, ?, ?)"),
        (username, user_input.strip(), bot_reply.strip())
    )
    conn.commit()
    conn.close()

    # Also write to local file in SQLite dev mode (keeps dataset.html working)
    if not database._USE_POSTGRES:
        data_path = get_user_data_path(username)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "a", encoding="utf-8") as f:
            f.write(f"{user_input.strip()} → {bot_reply.strip()}\n")


def write_pairs_to_user_data(username, pairs):
    """Bulk-insert (input, output) tuples from onboarding."""
    conn = database.get_conn()
    c = conn.cursor()
    for inp, out in pairs:
        c.execute(
            database._pg("INSERT INTO user_chat_data (username, input, output) VALUES (?, ?, ?)"),
            (username, inp.strip(), out.strip())
        )
    conn.commit()
    conn.close()

    if not database._USE_POSTGRES:
        data_path = get_user_data_path(username)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "a", encoding="utf-8") as f:
            for inp, out in pairs:
                f.write(f"{inp.strip()} → {out.strip()}\n")


def count_user_data(username):
    """Count conversation pairs stored for this user."""
    conn = database.get_conn()
    c = conn.cursor()
    c.execute(
        database._pg("SELECT COUNT(*) FROM user_chat_data WHERE username = ?"),
        (username,)
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0


def _get_recent_pairs(username, limit=40):
    """Fetch the most recent N chat pairs for Gemini context."""
    conn = database.get_conn()
    c = conn.cursor()
    c.execute(
        database._pg(
            "SELECT input, output FROM user_chat_data "
            "WHERE username = ? ORDER BY id DESC LIMIT ?"
        ),
        (username, limit)
    )
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))   # oldest first → better prompt order


# ── Training (becomes a lightweight no-op for Gemini mode) ───────────────────

def train_user_model(username):
    """
    In Gemini mode: data is already in DB, nothing to train.
    In local SQLite dev mode: rebuild .pkl from local file if it exists.
    """
    if database._USE_POSTGRES or not GEMINI_KEY:
        count = count_user_data(username)
        if count < 3:
            return False, f"Need at least 3 conversation pairs. You have {count}."
        return True, f"Twin ready — {count} conversation pairs loaded. 🧠"

    # Local fallback: try the legacy pkl approach
    data_path = get_user_data_path(username)
    if not os.path.exists(data_path):
        return False, "No training data found."
    count = count_user_data(username)
    if count < 3:
        return False, f"Need at least 3 pairs. You have {count}."
    return True, f"Twin trained on {count} conversation pairs."


def invalidate_user_model(username):
    pass   # nothing to invalidate — Gemini reads DB live


def learn_from_chat(username, user_input, bot_reply):
    """Store new pair in DB (already done by add_to_user_data)."""
    pass


def learn_from_chat_background(username, user_input, bot_reply):
    """Sync in serverless; data written by add_to_user_data already."""
    pass


# ── Decision feedback bridge (kept for cross-module compat) ──────────────────

CHAT_DECISION_SIM_THRESHOLD = 0.38


def search_decision_feedback(username, user_input):
    """Look for confirmed past decisions similar to the chat message."""
    try:
        from decision_utils import get_confirmed_training_data
        rows = get_confirmed_training_data(username)
        if not rows:
            return None, 0.0

        query = user_input.lower().strip()
        for ctx, opt_a, opt_b, correct in rows:
            combined = f"{ctx} {opt_a} {opt_b}".lower()
            # Simple keyword overlap score (no sklearn on Vercel)
            q_words   = set(query.split())
            c_words   = set(combined.split())
            overlap   = len(q_words & c_words)
            score     = overlap / max(len(q_words), 1)
            if score >= CHAT_DECISION_SIM_THRESHOLD:
                answer = (
                    f"Based on your past choices, you prefer **{correct}** "
                    f"when it comes to '{ctx}'. "
                    f"You've weighed {opt_a} vs {opt_b} before and went with {correct}. 🎯"
                )
                return answer, round(min(score * 105, 96.0), 1)
    except Exception:
        pass
    return None, 0.0


# ── Core reply function ───────────────────────────────────────────────────────

def chat_reply(user_input, username=None):
    """
    Generate a reply as the user's digital twin.

    Priority:
      1. Decision-feedback history match (personalised past choice)
      2. Gemini API with recent conversation pairs as few-shot context
      3. Keyword fallback (when Gemini key is absent — local dev)
    """
    if not user_input.strip():
        return "Say something! 😄", 0.0

    # ── 1. Decision feedback override ────────────────────────────────────────
    if username:
        dec_reply, dec_conf = search_decision_feedback(username, user_input)
        if dec_reply:
            return dec_reply, dec_conf

    # ── 2. Gemini with conversation history ──────────────────────────────────
    model = _get_model()
    if model and username:
        pairs = _get_recent_pairs(username, limit=40)
        if pairs:
            examples = "\n".join(
                f"User said: \"{inp}\"\nTwin replied: \"{out}\""
                for inp, out in pairs[-20:]   # last 20 for context window
            )
            prompt = (
                f"You are the digital twin of user '{username}'. "
                f"Your job is to reply exactly as that person would — "
                f"using their tone, vocabulary, and personality. "
                f"Here are examples of how they talk:\n\n"
                f"{examples}\n\n"
                f"Now the user says: \"{user_input}\"\n"
                f"Reply as their twin (1-3 sentences, matching their style):"
            )
            try:
                response = model.generate_content(prompt)
                reply = response.text.strip()
                return reply, 0.92
            except Exception as e:
                return f"Twin is thinking... (error: {e})", 0.0
        else:
            # No history yet — ask Gemini for a friendly default
            try:
                response = model.generate_content(
                    f"You are a digital twin AI. The user '{username}' just said: \"{user_input}\". "
                    f"Reply in a friendly, casual way (1-2 sentences):"
                )
                return response.text.strip(), 0.75
            except Exception:
                pass

    # ── 3. Keyword fallback (local dev without Gemini key) ───────────────────
    if username:
        pairs = _get_recent_pairs(username, limit=50)
        if pairs:
            query   = user_input.lower()
            best    = None
            best_sc = 0
            for inp, out in pairs:
                words  = set(inp.lower().split())
                q_words = set(query.split())
                sc     = len(words & q_words) / max(len(q_words), 1)
                if sc > best_sc:
                    best_sc = sc
                    best    = out
            if best and best_sc >= 0.35:
                return best, best_sc

    return "My twin hasn't been trained yet — chat more to teach me! 🤖", 0.0