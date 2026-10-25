import sqlite3
from datetime import datetime

DB_NAME = "tb_vision.db"

def init_db():
    """Create the predictions logging table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_logs (
            scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            true_label TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL,
            status TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(filename, true_label, predicted_label, confidence, status):
    """Save an inference record to the database."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO scan_logs (filename, true_label, predicted_label, confidence, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, true_label, predicted_label, confidence, status, current_time))
    
    conn.commit()
    conn.close()

def get_all_logs():
    """Fetch all saved inference history from the database."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT scan_id, filename, true_label, predicted_label, confidence, status, timestamp FROM scan_logs")
    records = cursor.fetchall()
    conn.close()
    return records
