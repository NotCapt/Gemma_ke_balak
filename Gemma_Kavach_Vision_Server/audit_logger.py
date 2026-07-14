import sqlite3
import json
import os
from datetime import datetime

# Initialize SQLite database path
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db"))
DB_PATH = os.path.join(DB_DIR, "kavach.db")

def init_db():
    """Create database tables if they don't exist"""
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        location TEXT,
        operator_name TEXT,
        status TEXT,
        created_at TEXT,
        last_analysis TEXT,
        frames_analyzed INTEGER,
        frames_flagged INTEGER,
        risk_score REAL,
        analysis_breakdown TEXT
    )
    """)
    
    # Frame Analyses table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS frame_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        frame_number INTEGER,
        crowd_density TEXT,
        crowd_motion TEXT,
        risk_level TEXT,
        risk_detected BOOLEAN,
        timestamp TEXT,
        image_path TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
    )
    """)
    
    # Audit Trail table (Append-only forensic log)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_trail (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        timestamp TEXT,
        event_type TEXT,
        severity TEXT,
        reasoning TEXT,
        action_taken TEXT,
        metadata TEXT
    )
    """)
    
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()

def save_session_to_db(session_id: str, session_data: dict) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        breakdown = json.dumps(session_data.get("analysis_breakdown", {}))
        
        cursor.execute("""
        INSERT OR REPLACE INTO sessions 
        (session_id, location, operator_name, status, created_at, last_analysis, frames_analyzed, frames_flagged, risk_score, analysis_breakdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            session_data.get("location"),
            session_data.get("operator_name"),
            session_data.get("status"),
            session_data.get("created_at"),
            session_data.get("last_analysis"),
            session_data.get("frames_analyzed", 0),
            session_data.get("frames_flagged", 0),
            session_data.get("risk_score", 0.0),
            breakdown
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ DB save error: {e}")
        return False

def load_session_from_db(session_id: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
            
        session_data = dict(row)
        if session_data["analysis_breakdown"]:
            session_data["analysis_breakdown"] = json.loads(session_data["analysis_breakdown"])
            
        # Load flagged frames manually since we don't store them directly in the sessions table JSON anymore
        cursor.execute("SELECT * FROM frame_analyses WHERE session_id = ? AND risk_detected = 1 ORDER BY frame_number ASC", (session_id,))
        flagged = cursor.fetchall()
        
        flagged_frames = []
        for f in flagged:
            flagged_frames.append({
                "frame_number": f["frame_number"],
                "crowd_density": f["crowd_density"],
                "crowd_motion": f["crowd_motion"],
                "risk_level": f["risk_level"],
                "timestamp": f["timestamp"],
                "local_path": f["image_path"]
            })
            
        session_data["flagged_frames"] = flagged_frames
        
        conn.close()
        return session_data
    except Exception as e:
        print(f"❌ DB load error: {e}")
        return None

def log_frame_analysis(session_id: str, frame_number: int, density: str, motion: str, risk_level: str, risk_detected: bool, image_path: str = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
        INSERT INTO frame_analyses 
        (session_id, frame_number, crowd_density, crowd_motion, risk_level, risk_detected, timestamp, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, frame_number, density, motion, risk_level, risk_detected, timestamp, image_path))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging frame analysis: {e}")

def log_audit_trail(session_id: str, event_type: str, severity: str, reasoning: str, action_taken: str, metadata: dict = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        cursor.execute("""
        INSERT INTO audit_trail 
        (session_id, timestamp, event_type, severity, reasoning, action_taken, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, timestamp, event_type, severity, reasoning, action_taken, meta_str))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging audit trail: {e}")
