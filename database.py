"""
database.py — Dual-mode database layer.

- Local development : SQLite  (default, no setup needed)
- Vercel production  : Postgres via DATABASE_URL environment variable

All other modules should call get_conn() and use the returned connection object.
The connection's placeholder style (? for SQLite, %s for Postgres) is abstracted
by the `ph()` helper so SQL strings stay portable.
"""

import os
import sqlite3

# When DATABASE_URL is set (Vercel / Neon), switch to Postgres
DATABASE_URL = os.environ.get("DATABASE_URL", "")
_USE_POSTGRES = bool(DATABASE_URL)

if _USE_POSTGRES:
    import psycopg2
    import psycopg2.extras


def get_conn():
    """Return a live database connection (SQLite or Postgres)."""
    if _USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    return sqlite3.connect("app.db")


def ph(n=1):
    """
    Return the correct SQL placeholder string for n params.
    SQLite uses ?   → ph(3) → "?, ?, ?"
    Postgres uses %s → ph(3) → "%s, %s, %s"
    """
    mark = "%s" if _USE_POSTGRES else "?"
    return ", ".join([mark] * n)


def _pg(sql):
    """Convert SQLite-style ? placeholders to Postgres %s."""
    if _USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql


def fetchone(cursor):
    return cursor.fetchone()


def fetchall(cursor):
    return cursor.fetchall()


def lastrowid(cursor):
    if _USE_POSTGRES:
        return cursor.fetchone()[0]
    return cursor.lastrowid


def init_db():
    conn = get_conn()
    c = conn.cursor()

    if _USE_POSTGRES:
        c.execute("""
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        SERIAL PRIMARY KEY,
                username  TEXT,
                sender    TEXT,
                message   TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_chat_data (
                id        SERIAL PRIMARY KEY,
                username  TEXT NOT NULL,
                input     TEXT NOT NULL,
                output    TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT UNIQUE NOT NULL,
                password    TEXT NOT NULL,
                name        TEXT,
                bio         TEXT,
                avatar      TEXT,
                personality TEXT DEFAULT 'casual'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT,
                sender    TEXT,
                message   TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS decision_feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT NOT NULL,
                context     TEXT NOT NULL,
                option_a    TEXT NOT NULL,
                option_b    TEXT NOT NULL,
                predicted   TEXT NOT NULL,
                correct     TEXT,
                was_wrong   INTEGER DEFAULT 0,
                reason      TEXT DEFAULT '',
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_chat_data (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT NOT NULL,
                input     TEXT NOT NULL,
                output    TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    conn.commit()
    conn.close()