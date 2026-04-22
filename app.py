from flask import Flask, render_template, request, jsonify, redirect, session
from chat_utils import (
    chat_reply, train_user_model, ensure_user_data_file,
    add_to_user_data, count_user_data, write_pairs_to_user_data,
    get_user_data_path, get_user_model_path
)
from decision_utils import predict
from personality import apply_style
from onboarding_questions import QUESTIONS, get_training_pairs_from_answers
import database
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "secret123"

database.init_db()


def db():
    return sqlite3.connect("app.db")


# ─── AUTH ─────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()

        conn = db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = u
            ensure_user_data_file(u)
            return redirect("/")
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html", error="")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u           = request.form.get("username", "").strip()
        p           = request.form.get("password", "").strip()
        name        = request.form.get("name", "").strip()
        bio         = request.form.get("bio", "").strip()
        personality = request.form.get("personality", "casual")
        avatar      = f"https://i.pravatar.cc/150?u={u}"

        if not u or not p or not name:
            return render_template("signup.html", error="Username, password and name are required")

        conn = db()
        c = conn.cursor()
        try:
            c.execute("""
                INSERT INTO users(username, password, name, bio, avatar, personality)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (u, p, name, bio, avatar, personality))
            conn.commit()
        except Exception:
            conn.close()
            return render_template("signup.html", error="Username already exists")
        conn.close()

        # Create empty data file (onboarding will fill it)
        ensure_user_data_file(u, seed_data=False)

        session["user"] = u
        # Flag: needs onboarding
        session["needs_onboarding"] = True

        return redirect("/onboarding")
    return render_template("signup.html", error="")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ─── ONBOARDING ───────────────────────────────────────────────────

@app.route("/onboarding")
def onboarding():
    if "user" not in session:
        return redirect("/login")
    return render_template("onboarding.html",
                           questions=QUESTIONS,
                           total=len(QUESTIONS))


@app.route("/onboarding/save", methods=["POST"])
def onboarding_save():
    """Receive all answers, generate pairs, train model."""
    if "user" not in session:
        return jsonify({"success": False, "message": "Not logged in"})

    username = session["user"]
    answers  = request.json  # { question_id: answer_text }

    if not answers:
        return jsonify({"success": False, "message": "No answers received"})

    # Generate training pairs from answers
    pairs = get_training_pairs_from_answers(answers)

    if len(pairs) == 0:
        return jsonify({"success": False, "message": "No valid answers to train on"})

    # Write pairs to user's dataset
    write_pairs_to_user_data(username, pairs)

    # Train model
    success, message = train_user_model(username)

    # Clear onboarding flag
    session.pop("needs_onboarding", None)

    return jsonify({
        "success": success,
        "message": message,
        "pair_count": len(pairs)
    })


# ─── HOME ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")

    # Redirect to onboarding if not completed
    if session.get("needs_onboarding"):
        return redirect("/onboarding")

    conn = db()
    c = conn.cursor()
    c.execute("SELECT name, bio, avatar FROM users WHERE username=?", (session["user"],))
    user = c.fetchone()
    conn.close()

    if user is None:
        session.clear()
        return redirect("/login")

    ensure_user_data_file(session["user"])
    pair_count = count_user_data(session["user"])
    has_model  = os.path.exists(get_user_model_path(session["user"]))

    return render_template("index.html",
                           name=user[0],
                           bio=user[1],
                           avatar=user[2],
                           pair_count=pair_count,
                           has_model=has_model)


# ─── PROFILE ──────────────────────────────────────────────────────

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect("/login")

    conn = db()
    c = conn.cursor()

    if request.method == "POST":
        name        = request.form.get("name", "").strip() or session["user"]
        bio         = request.form.get("bio", "").strip()
        personality = request.form.get("personality", "casual")
        avatar      = request.form.get("avatar", "").strip() or f"https://i.pravatar.cc/150?u={session['user']}"

        c.execute("UPDATE users SET name=?, bio=?, personality=?, avatar=? WHERE username=?",
                  (name, bio, personality, avatar, session["user"]))
        conn.commit()
        conn.close()
        return redirect("/")

    c.execute("SELECT name, bio, avatar, personality FROM users WHERE username=?", (session["user"],))
    user = c.fetchone()
    conn.close()

    if user is None:
        session.clear()
        return redirect("/login")

    return render_template("profile.html",
                           name=user[0], bio=user[1],
                           avatar=user[2], personality=user[3])


# ─── DATASET ──────────────────────────────────────────────────────

@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    if "user" not in session:
        return redirect("/login")

    username  = session["user"]
    data_path = get_user_data_path(username)
    ensure_user_data_file(username)
    message = ""

    if request.method == "POST":
        action = request.form.get("action")

        if action == "save":
            content = request.form.get("content", "")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(content)
            message = "✅ Dataset saved."

        elif action == "add_pair":
            inp = request.form.get("inp", "").strip()
            out = request.form.get("out", "").strip()
            if inp and out:
                with open(data_path, "a", encoding="utf-8") as f:
                    f.write(f"{inp} → {out}\n")
                message = f"✅ Added: '{inp} → {out}'"
            else:
                message = "⚠ Both fields are required."

        elif action == "train":
            success, msg = train_user_model(username)
            message = ("✅ " if success else "❌ ") + msg

    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    return render_template("dataset.html",
                           content=content,
                           pair_count=count_user_data(username),
                           has_model=os.path.exists(get_user_model_path(username)),
                           message=message)


# ─── TRAIN API ────────────────────────────────────────────────────

@app.route("/train", methods=["POST"])
def train():
    if "user" not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    username       = session["user"]
    success, message = train_user_model(username)
    return jsonify({"success": success, "message": message,
                    "pair_count": count_user_data(username)})


# ─── CHAT ─────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"reply": "Please log in first.", "confidence": 0})

    data = request.json
    if not data or "message" not in data:
        return jsonify({"reply": "No message received.", "confidence": 0})

    msg      = data["message"]
    username = session["user"]

    conn = db()
    c = conn.cursor()
    c.execute("SELECT personality FROM users WHERE username=?", (username,))
    row = c.fetchone()
    personality = row[0] if row else "casual"

    reply, conf = chat_reply(msg, username=username)
    styled      = apply_style(reply, personality)

    c.execute("INSERT INTO messages VALUES (NULL,?,?,?,CURRENT_TIMESTAMP)", (username, "user", msg))
    c.execute("INSERT INTO messages VALUES (NULL,?,?,?,CURRENT_TIMESTAMP)", (username, "bot", styled))
    conn.commit()
    conn.close()

    add_to_user_data(username, msg, reply)

    return jsonify({"reply": styled, "confidence": round(conf * 100, 1)})


# ─── DECIDE ───────────────────────────────────────────────────────

@app.route("/decide", methods=["POST"])
def decide():
    if "user" not in session:
        return jsonify({"decision": "Please log in.", "confidence": 0, "reason": ""})

    data = request.json or {}
    ctx  = data.get("context", "")
    a    = data.get("a", "")
    b    = data.get("b", "")

    if not ctx or not a or not b:
        return jsonify({"decision": "Fill all fields.", "confidence": 0, "reason": ""})

    decision, conf, reason = predict(ctx, a, b)
    return jsonify({"decision": decision, "confidence": conf, "reason": reason})


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)