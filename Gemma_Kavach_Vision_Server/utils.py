# utils.py - Utility functions for dual crowd analysis (local-first, no cloud dependency)
import json
import uuid
import time
import smtplib
import os
from email.message import EmailMessage
from typing import Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Email Configuration
EMAIL_SENDER = "yorgashimal2008@gmail.com"
EMAIL_PASSWORD = os.getenv("GOOGLE_APP_PASSWORD")
EMAIL_RECEIVER = "yorgashimal2008@gmail.com"

# Enhanced Alert thresholds
MIN_FRAMES_FOR_ALERT = 5
RISK_THRESHOLD_FOR_ALERT = 70.0  # Increased threshold for more sophisticated scoring
CRITICAL_FRAMES_THRESHOLD = 2    # Send alert if 2+ CRITICAL frames detected

# Risk scoring weights
RISK_WEIGHTS = {
    "SAFE": 0,
    "MODERATE": 25,
    "HIGH": 60,
    "CRITICAL": 100
}

from audit_logger import (
    save_session_to_db, 
    load_session_from_db, 
    log_frame_analysis, 
    log_audit_trail
)

# ---------------------------------------------------------------------------
# Session helpers — thin wrappers over SQLite (replaces old GCS functions)
# ---------------------------------------------------------------------------

def save_session(session_id: str, session_data: Dict[str, Any]) -> bool:
    return save_session_to_db(session_id, session_data)

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    return load_session_from_db(session_id)

# Backward-compatible aliases (used by older callers)
save_session_to_gcs = save_session
load_session_from_gcs = load_session


def calculate_risk_score(session_data: Dict[str, Any]) -> float:
    """
    Calculate sophisticated risk score based on analysis breakdown.
    Considers both frequency of risks and severity levels.
    """
    try:
        frames_analyzed = session_data.get("frames_analyzed", 0)
        if frames_analyzed == 0:
            return 0.0
            
        breakdown = session_data.get("analysis_breakdown", {})
        risk_levels = breakdown.get("risk_levels", {})
        
        # Calculate weighted score
        total_weighted_score = 0
        for risk_level, count in risk_levels.items():
            weight = RISK_WEIGHTS.get(risk_level, 0)
            total_weighted_score += (count * weight)
        
        # Average weighted score
        average_score = total_weighted_score / frames_analyzed
        
        # Apply additional penalties for concerning patterns
        density_stats = breakdown.get("density_stats", {})
        motion_stats = breakdown.get("motion_stats", {})
        
        # Penalty for high density frames
        high_density_ratio = density_stats.get("High", 0) / frames_analyzed
        if high_density_ratio > 0.3:  # More than 30% high density
            average_score *= 1.2
            
        # Penalty for chaotic motion
        chaotic_ratio = motion_stats.get("Chaotic", 0) / frames_analyzed
        if chaotic_ratio > 0.2:  # More than 20% chaotic
            average_score *= 1.3
            
        # Cap at 100
        return min(round(average_score, 2), 100.0)
        
    except Exception as e:
        print(f"❌ Error calculating risk score: {e}")
        # Fallback to simple calculation
        frames_flagged = session_data.get("frames_flagged", 0)
        frames_analyzed = session_data.get("frames_analyzed", 1)
        return round((frames_flagged / frames_analyzed) * 100, 2) if frames_analyzed > 0 else 0.0

def get_verdict(risk_score: float) -> str:
    """Convert risk score to verdict with enhanced thresholds"""
    if risk_score <= 15:
        return "SAFE"
    elif risk_score <= 40:
        return "WATCH"
    elif risk_score <= 70:
        return "ALERT"
    else:
        return "CRITICAL"

def should_send_alert(session_data: Dict[str, Any]) -> bool:
    """
    Enhanced alert logic considering multiple factors
    """
    frames_analyzed = session_data.get("frames_analyzed", 0)
    risk_score = session_data.get("risk_score", 0.0)
    email_sent = session_data.get("email_sent", False)
    
    if email_sent or frames_analyzed < MIN_FRAMES_FOR_ALERT:
        return False
    
    # Check for high risk score
    if risk_score >= RISK_THRESHOLD_FOR_ALERT:
        return True
    
    # Check for critical frames
    breakdown = session_data.get("analysis_breakdown", {})
    critical_frames = breakdown.get("risk_levels", {}).get("CRITICAL", 0)
    if critical_frames >= CRITICAL_FRAMES_THRESHOLD:
        print(f"🚨 Alert triggered by {critical_frames} CRITICAL frames")
        return True
    
    # Check for concerning patterns (e.g., rapid increase in risk)
    flagged_frames = session_data.get("flagged_frames", [])
    if len(flagged_frames) >= 3:
        # Check if last 3 frames were all flagged (rapid escalation)
        recent_frames = flagged_frames[-3:]
        frame_numbers = [f["frame_number"] for f in recent_frames]
        if len(frame_numbers) == 3 and max(frame_numbers) - min(frame_numbers) <= 2:
            print(f"🚨 Alert triggered by rapid escalation pattern")
            return True
    
    return False

def send_alert_email(session_id: str, session_data: Dict[str, Any]) -> bool:
    """Enhanced alert email with detailed analysis breakdown"""
    try:
        if not EMAIL_PASSWORD:
            print("⚠️ Email password not set. Skipping email notification.")
            return False
            
        print(f"📧 Sending enhanced alert email for session {session_id}...")
        
        # Create email message
        msg = EmailMessage()
        msg["Subject"] = f"🚨 CROWD SAFETY ALERT - {session_data['location']} (Session {session_id})"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        
        # Get analysis breakdown
        breakdown = session_data.get("analysis_breakdown", {})
        density_stats = breakdown.get("density_stats", {})
        motion_stats = breakdown.get("motion_stats", {})
        risk_levels = breakdown.get("risk_levels", {})
        
        # Build enhanced email body
        body_lines = [
            f"🚨 CROWD SAFETY ALERT 🚨",
            f"",
            f"📍 Location: {session_data['location']}",
            f"👤 Operator: {session_data['operator_name']}",
            f"🆔 Session ID: {session_id}",
            f"⚠️ Verdict: {get_verdict(session_data['risk_score'])}",
            f"📊 Risk Score: {session_data['risk_score']}%",
            f"🕐 Alert Time: {get_current_timestamp()}",
            f"",
            f"📈 ANALYSIS SUMMARY:",
            f"├── Total Frames Analyzed: {session_data['frames_analyzed']}",
            f"├── Flagged Frames: {session_data['frames_flagged']}",
            f"└── Flagging Rate: {round((session_data['frames_flagged'] / session_data['frames_analyzed']) * 100, 1)}%",
            f"",
            f"👥 CROWD DENSITY BREAKDOWN:",
            f"├── High Density: {density_stats.get('High', 0)} frames",
            f"├── Medium Density: {density_stats.get('Medium', 0)} frames",
            f"└── Low Density: {density_stats.get('Low', 0)} frames",
            f"",
            f"🏃 CROWD MOTION BREAKDOWN:",
            f"├── Chaotic Motion: {motion_stats.get('Chaotic', 0)} frames",
            f"└── Calm Motion: {motion_stats.get('Calm', 0)} frames",
            f"",
            f"🚦 RISK LEVEL BREAKDOWN:",
            f"├── CRITICAL: {risk_levels.get('CRITICAL', 0)} frames",
            f"├── HIGH: {risk_levels.get('HIGH', 0)} frames",
            f"├── MODERATE: {risk_levels.get('MODERATE', 0)} frames",
            f"└── SAFE: {risk_levels.get('SAFE', 0)} frames",
            f"",
        ]
        
        # Add flagged frame details with reasoning
        flagged_frames = session_data.get("flagged_frames", [])
        if flagged_frames:
            body_lines.append("🔍 FLAGGED FRAME DETAILS:")
            for i, frame_info in enumerate(flagged_frames[-5:], 1):  # Show last 5
                reasoning = frame_info.get("reasoning", {})
                reasoning_text = reasoning.get("reasoning", "") if isinstance(reasoning, dict) else str(reasoning)
                body_lines.append(
                    f"{i}. Frame {frame_info['frame_number']}: "
                    f"{frame_info['risk_level']} "
                    f"(Density: {frame_info.get('crowd_density', 'Unknown')}, "
                    f"Motion: {frame_info.get('crowd_motion', 'Unknown')}) "
                    f"@ {frame_info['timestamp']}"
                )
                if reasoning_text:
                    body_lines.append(f"   Reasoning: {reasoning_text[:200]}")
            body_lines.append("")
        
        body_lines.extend([
            f"⚠️ IMMEDIATE ACTION REQUIRED!",
            f"",
            f"Please investigate the situation immediately and take appropriate crowd control measures.",
            f"",
            f"📂 Session Data: Available in local database (Session {session_id})",
            f"🔍 Audit Trail: GET /api/audit/{session_id}",
            f"",
            f"This alert was generated by Gemma Kavach Vision System (on-device inference)"
        ])
        
        msg.set_content("\n".join(body_lines))
        
        # Attach recent flagged images
        attachment_count = 0
        for frame_info in flagged_frames[-3:]:  # Last 3 flagged images
            local_path = frame_info.get("local_path")
            if local_path and os.path.exists(local_path):
                with open(local_path, "rb") as img_file:
                    image_data = img_file.read()
                    risk_level = frame_info.get('risk_level', 'FLAGGED')
                    filename = f"{risk_level}_frame_{frame_info['frame_number']:03d}.jpg"
                    msg.add_attachment(
                        image_data,
                        maintype="image",
                        subtype="jpeg", 
                        filename=filename
                    )
                    attachment_count += 1
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
            
        print(f"✅ Enhanced alert email sent! ({attachment_count} images attached)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send alert email: {e}")
        return False

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())[:8]

def get_current_timestamp() -> str:
    """Get current timestamp in standard format"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def save_flagged_image(session_id: str, frame_number: int, image_content: bytes) -> Optional[str]:
    """Save flagged image locally and return the path"""
    try:
        dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "flagged_frames", session_id))
        os.makedirs(dir_path, exist_ok=True)
        
        file_path = os.path.join(dir_path, f"frame_{frame_number:03d}.jpg")
        with open(file_path, "wb") as f:
            f.write(image_content)
            
        return file_path
        
    except Exception as e:
        print(f"❌ Error saving flagged image locally: {e}")
        return None

# Backward-compatible alias
save_flagged_image_to_gcs = save_flagged_image

def check_db_connection() -> Dict[str, str]:
    """Check Database connection status"""
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "kavach.db"))
    
    return {
        "database": "SQLite",
        "status": "connected" if os.path.exists(db_path) else "disconnected",
        "path": db_path,
        "analysis_type": "dual_crowd_analysis"
    }

# Backward-compatible alias
check_gcs_connection = check_db_connection