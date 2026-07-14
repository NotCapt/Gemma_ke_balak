import sqlite3
import os

db = os.path.join("db", "kavach.db")
conn = sqlite3.connect(db)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    risk_level TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    evidence TEXT,
    frame_refs TEXT,
    email_sent BOOLEAN DEFAULT 0,
    created_at TEXT
)""")

c.execute("""CREATE TABLE IF NOT EXISTS emergency_reports (
    report_id TEXT PRIMARY KEY,
    message TEXT NOT NULL,
    classification TEXT NOT NULL,
    severity TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    recommended_action TEXT,
    location TEXT,
    contact TEXT,
    image_path TEXT,
    created_at TEXT
)""")

# Ensure audit_trail has report_id column
try:
    c.execute("ALTER TABLE audit_trail ADD COLUMN report_id TEXT")
    print("Added report_id column to audit_trail")
except:
    print("report_id column already exists in audit_trail")

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in c.fetchall()])
conn.commit()
conn.close()
print("Done")
