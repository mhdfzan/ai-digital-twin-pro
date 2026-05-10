"""
migrate_to_neon.py — One-time migration from local SQLite → Neon Postgres.

Run this ONCE after setting up your Neon database:

    pip install psycopg2-binary
    set DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
    python migrate_to_neon.py

What it migrates:
  - users          (accounts, passwords, profiles)
  - messages       (chat history)
  - decision_feedback (rated decisions)
  - user_chat_data  (training conversation pairs, from local .txt files)
"""

import sqlite3
import os
import sys

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    print("[ERROR] Set DATABASE_URL environment variable first.")
    print("    Example: set DATABASE_URL=postgresql://user:pass@host/db?sslmode=require")
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("[ERROR] psycopg2-binary not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

print("[*] Connecting to Neon Postgres...")
pg = psycopg2.connect(DATABASE_URL)
pg_c = pg.cursor()

print("[*] Opening local SQLite app.db...")
sq = sqlite3.connect("app.db")
sq_c = sq.cursor()

# ── Create tables in Postgres ─────────────────────────────────────────────────

print("[*] Creating tables in Postgres...")
pg_c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id          SERIAL PRIMARY KEY,
        username    TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,
        name        TEXT,
        bio         TEXT,
        avatar      TEXT,
        personality TEXT DEFAULT 'casual'
    )
""")
pg_c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id        SERIAL PRIMARY KEY,
        username  TEXT,
        sender    TEXT,
        message   TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
pg_c.execute("""
    CREATE TABLE IF NOT EXISTS decision_feedback (
        id          SERIAL PRIMARY KEY,
        username    TEXT NOT NULL,
        context     TEXT NOT NULL,
        option_a    TEXT NOT NULL,
        option_b    TEXT NOT NULL,
        predicted   TEXT NOT NULL,
        correct     TEXT,
        was_wrong   INTEGER DEFAULT 0,
        reason      TEXT DEFAULT '',
        timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
pg_c.execute("""
    CREATE TABLE IF NOT EXISTS user_chat_data (
        id        SERIAL PRIMARY KEY,
        username  TEXT NOT NULL,
        input     TEXT NOT NULL,
        output    TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
pg.commit()

# ── Migrate users ─────────────────────────────────────────────────────────────

sq_c.execute("SELECT username, password, name, bio, avatar, personality FROM users")
users = sq_c.fetchall()
print(f"[*] Migrating {len(users)} user(s)...")
for u in users:
    try:
        pg_c.execute(
            "INSERT INTO users (username, password, name, bio, avatar, personality) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
            u
        )
    except Exception as e:
        print(f"   [!] Skipped user {u[0]}: {e}")
pg.commit()

# ── Migrate messages ──────────────────────────────────────────────────────────

sq_c.execute("SELECT username, sender, message, timestamp FROM messages")
msgs = sq_c.fetchall()
print(f"[*] Migrating {len(msgs)} message(s)...")
for m in msgs:
    pg_c.execute(
        "INSERT INTO messages (username, sender, message, timestamp) VALUES (%s, %s, %s, %s)",
        m
    )
pg.commit()

# ── Migrate decision_feedback ─────────────────────────────────────────────────

sq_c.execute(
    "SELECT username, context, option_a, option_b, predicted, correct, was_wrong, reason, timestamp "
    "FROM decision_feedback"
)
decisions = sq_c.fetchall()
print(f"[*] Migrating {len(decisions)} decision record(s)...")
for d in decisions:
    pg_c.execute(
        "INSERT INTO decision_feedback "
        "(username, context, option_a, option_b, predicted, correct, was_wrong, reason, timestamp) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        d
    )
pg.commit()

# ── Migrate chat pairs from local .txt files ──────────────────────────────────

data_root = os.path.join("data", "users")
chat_pairs_count = 0

if os.path.exists(data_root):
    for username in os.listdir(data_root):
        txt_path = os.path.join(data_root, username, "user_data.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "→" not in line:
                    continue
                parts = line.split("→", 1)
                if len(parts) != 2:
                    continue
                inp, out = parts[0].strip(), parts[1].strip()
                if inp and out:
                    pg_c.execute(
                        "INSERT INTO user_chat_data (username, input, output) VALUES (%s, %s, %s)",
                        (username, inp, out)
                    )
                    chat_pairs_count += 1
    pg.commit()

print(f"[*] Migrated {chat_pairs_count} chat pair(s) from local .txt files.")

# ── Done ──────────────────────────────────────────────────────────────────────

sq.close()
pg.close()

print()
print("[OK] Migration complete!")
print("     All your local data is now in Neon Postgres.")
print("     You can now deploy to Vercel.")
