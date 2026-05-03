import sqlite3


def get_conn():
    return sqlite3.connect("app.db")


def init_db():
    conn = get_conn()
    c = conn.cursor()

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

    # Feedback-driven ML learning table
    # Stores every prediction + the user's rating so the model can learn from mistakes
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

    conn.commit()
    conn.close()