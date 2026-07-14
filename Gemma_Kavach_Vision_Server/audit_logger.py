import sqlite3
import json
import os
from datetime import datetime

# Initialize SQLite database path
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db"))
DB_PATH = os.path.join(DB_DIR, "kavach.db")

def _get_connection():
    """Get a new SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create database tables if they don't exist"""
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = _get_connection()
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
    
    # Alerts table — triggered alerts with reasoning (design spec)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT REFERENCES sessions(session_id),
        risk_level TEXT NOT NULL,
        reasoning TEXT NOT NULL,
        evidence TEXT,
        frame_refs TEXT,
        email_sent BOOLEAN DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)
    
    # Emergency reports with classification + reasoning (design spec)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emergency_reports (
        report_id TEXT PRIMARY KEY,
        message TEXT NOT NULL,
        classification TEXT NOT NULL,
        severity TEXT NOT NULL,
        reasoning TEXT NOT NULL,
        recommended_action TEXT,
        location TEXT,
        contact TEXT,
        image_path TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)
    
    # Audit Trail table (Append-only forensic log)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_trail (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        report_id TEXT,
        timestamp TEXT,
        event_type TEXT NOT NULL,
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


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def save_session_to_db(session_id: str, session_data: dict) -> bool:
    try:
        conn = _get_connection()
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
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
            
        session_data = dict(row)
        if session_data["analysis_breakdown"]:
            session_data["analysis_breakdown"] = json.loads(session_data["analysis_breakdown"])
            
        # Load flagged frames from frame_analyses table
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


# ---------------------------------------------------------------------------
# Frame analysis logging
# ---------------------------------------------------------------------------

def log_frame_analysis(session_id: str, frame_number: int, density: str, motion: str, risk_level: str, risk_detected: bool, image_path: str = None):
    try:
        conn = _get_connection()
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


# ---------------------------------------------------------------------------
# Alert logging (NEW — matches design spec)
# ---------------------------------------------------------------------------

def log_alert(session_id: str, risk_level: str, reasoning: str, evidence: dict = None, frame_refs: str = None, email_sent: bool = False) -> int:
    """Log an alert with reasoning to the alerts table. Returns the alert id."""
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        evidence_str = json.dumps(evidence) if evidence else None

        cursor.execute("""
        INSERT INTO alerts (session_id, risk_level, reasoning, evidence, frame_refs, email_sent, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, risk_level, reasoning, evidence_str, frame_refs, email_sent, timestamp))

        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return alert_id
    except Exception as e:
        print(f"❌ Error logging alert: {e}")
        return -1


# ---------------------------------------------------------------------------
# Emergency report logging (NEW — matches design spec)
# ---------------------------------------------------------------------------

def log_emergency_report(report_id: str, message: str, classification: str, severity: str, reasoning: str,
                         recommended_action: str = None, location: str = None, contact: str = None, image_path: str = None):
    """Persist an emergency report to the emergency_reports table."""
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO emergency_reports
        (report_id, message, classification, severity, reasoning, recommended_action, location, contact, image_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (report_id, message, classification, severity, reasoning,
              recommended_action, location, contact, image_path, datetime.now().isoformat()))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging emergency report: {e}")


# ---------------------------------------------------------------------------
# Audit trail (append-only forensic log)
# ---------------------------------------------------------------------------

def log_audit_trail(session_id: str = None, report_id: str = None, event_type: str = "system_event",
                    severity: str = None, reasoning: str = None, action_taken: str = None, metadata: dict = None):
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        cursor.execute("""
        INSERT INTO audit_trail 
        (session_id, report_id, timestamp, event_type, severity, reasoning, action_taken, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, report_id, timestamp, event_type, severity, reasoning, action_taken, meta_str))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging audit trail: {e}")


# ---------------------------------------------------------------------------
# Audit trail queries (read-only, for the audit API)
# ---------------------------------------------------------------------------

def get_full_audit_trail(session_id: str = None, limit: int = 100) -> list:
    """Return audit trail entries. Optionally filter by session_id."""
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                "SELECT * FROM audit_trail WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM audit_trail ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

        rows = cursor.fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            if entry.get("metadata"):
                try:
                    entry["metadata"] = json.loads(entry["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(entry)

        conn.close()
        return result
    except Exception as e:
        print(f"❌ Error reading audit trail: {e}")
        return []

def get_alerts(session_id: str = None, limit: int = 50) -> list:
    """Return alerts. Optionally filter by session_id."""
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                "SELECT * FROM alerts WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )

        rows = cursor.fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            if entry.get("evidence"):
                try:
                    entry["evidence"] = json.loads(entry["evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(entry)

        conn.close()
        return result
    except Exception as e:
        print(f"❌ Error reading alerts: {e}")
        return []

def get_emergency_reports(limit: int = 50) -> list:
    """Return recent emergency reports."""
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM emergency_reports ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        conn.close()
        return result
    except Exception as e:
        print(f"❌ Error reading emergency reports: {e}")
        return []
