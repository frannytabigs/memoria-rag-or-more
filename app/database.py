import sqlite3
import os

def get_db_connection():
    """Establishes a connection to the local SQLite file database."""
    # This will create the file if it doesn't exist
    db_path = os.path.join(os.path.dirname(__file__), '..', 'memoria_chat.db')
    
    # THE FIX: Add timeout=15 to tell SQLite to wait for locks to clear
    conn = sqlite3.connect(db_path, timeout=15)
    
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name like a dictionary
    return conn

def init_db():
    """Creates the chat_logs table if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()