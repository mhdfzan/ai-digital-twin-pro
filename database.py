import sqlite3

def get_conn():
    return sqlite3.connect("app.db")

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Users with profile fields
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT,
        bio TEXT,
        avatar TEXT,
        personality TEXT
    )
    """)

    # Messages (per user)
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        sender TEXT,
        message TEXT
    )
    """)

    conn.commit()
    conn.close()