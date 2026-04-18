from flask import Flask, render_template, request, jsonify, redirect, session
from chat_utils import chat_reply
from decision_utils import predict
from personality import apply_style
import database
import sqlite3

app = Flask(__name__)
app.secret_key = "secret123"

database.init_db()

def db():
    return sqlite3.connect("app.db")


# ---------- AUTH ----------

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        user = c.fetchone()

        if user:
            session["user"] = u
            return redirect("/")
        else:
            return "Invalid login"

    return render_template("login.html")


@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        name = request.form["name"]
        bio = request.form["bio"]

        avatar = "https://i.pravatar.cc/150?u=" + u
        personality = "casual"  # default

        conn = db()
        c = conn.cursor()

        try:
            c.execute("""
            INSERT INTO users(username,password,name,bio,avatar,personality)
            VALUES (?,?,?,?,?,?)
            """,(u,p,name,bio,avatar,personality))
            conn.commit()
        except:
            return "User already exists"

        return redirect("/login")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------- HOME ----------

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")

    conn = db()
    c = conn.cursor()
    c.execute("SELECT name, bio, avatar FROM users WHERE username=?",
              (session["user"],))
    user = c.fetchone()

    return render_template(
        "index.html",
        name=user[0],
        bio=user[1],
        avatar=user[2]
    )


# ---------- CHAT ----------

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["message"]
    user = session.get("user", "guest")

    # get personality
    conn = db()
    c = conn.cursor()
    c.execute("SELECT personality FROM users WHERE username=?", (user,))
    row = c.fetchone()
    personality = row[0] if row else "casual"

    # generate reply
    reply, _ = chat_reply(msg)
    reply = apply_style(reply, personality)

    # save history
    c.execute("INSERT INTO messages VALUES (NULL,?,?,?)",(user,"user",msg))
    c.execute("INSERT INTO messages VALUES (NULL,?,?,?)",(user,"bot",reply))
    conn.commit()

    return jsonify({"reply": reply})


# ---------- DECISION ----------

@app.route("/decide", methods=["POST"])
def decide():
    data = request.json

    decision, conf, reason = predict(
        data["context"],
        data["a"],
        data["b"]
    )

    return jsonify({
        "decision": decision,
        "confidence": conf,
        "reason": reason
    })


if __name__ == "__main__":
    app.run(debug=True)