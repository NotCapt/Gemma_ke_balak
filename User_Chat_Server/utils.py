# utils.py - Complete file with model loading + emergency functions
import os
import requests
import logging
import smtplib
import uuid
import time
from email.message import EmailMessage
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Categories
CATEGORIES = ["child_lost", "crowd_panic", "lost_item", "medical_help", "need_interpreter", "small_fire"]
CATEGORIES_STR = ", ".join(CATEGORIES)

# Email Configuration
EMAIL_SENDER = "yorgashimal2008@gmail.com"
EMAIL_PASSWORD = os.getenv("GOOGLE_APP_PASSWORD")
EMAIL_RECEIVER = "yorgashimal2008@gmail.com"

# Local Storage Configuration
EMERGENCY_IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "emergency_reports"))

GEMMA_API_URL = "http://localhost:8000/classify"

def load_model():
    """No-op: Model is now loaded centrally by GemmaServer"""
    logger.info("✅ Chat server now uses central GemmaServer for inference!")
    return True

def classify_emergency(text: str) -> dict:
    """Classify emergency text by calling central GemmaServer"""
    try:
        logger.info(f"Classifying text via GemmaServer: '{text}'")
        
        prompt = f"Classify this emergency into one of these categories: {CATEGORIES_STR}\n\nEmergency: {text}\n\nCategory:"
        
        payload = {
            "text": prompt,
            "max_tokens": 150
        }
        
        response = requests.post(GEMMA_API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # The API returns {category, severity, reasoning, action}
            # Map the returned category to one of our predefined ones if possible
            api_category = result.get("category", "").lower()
            mapped_category = "unknown"
            
            for cat in CATEGORIES:
                if cat in api_category:
                    mapped_category = cat
                    break
                    
            if mapped_category == "unknown":
                # Fallback mapping based on raw reasoning text if category not cleanly extracted
                reasoning = result.get("reasoning", "").lower()
                for cat in CATEGORIES:
                    if cat in reasoning:
                        mapped_category = cat
                        break
            
            result["classification"] = mapped_category
            return result
        else:
            logger.error(f"GemmaServer classification failed: {response.status_code}")
            return {"classification": "error", "reasoning": "API failed"}
            
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"classification": "error", "reasoning": str(e)}

def get_model_info():
    """Get model information"""
    return {
        "model_loaded": True,
        "categories": CATEGORIES,
        "model_path": "Delegated to GemmaServer",
        "device": "Delegated to GemmaServer"
    }

# Emergency Report Functions

def save_emergency_image(report_id: str, image_data: bytes, filename: str) -> Optional[str]:
    """Save emergency image locally and return the path"""
    try:
        os.makedirs(EMERGENCY_IMAGES_DIR, exist_ok=True)
        
        # Create unique filename
        file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
        local_filename = f"{report_id}_{int(time.time())}.{file_extension}"
        local_path = os.path.join(EMERGENCY_IMAGES_DIR, local_filename)
        
        with open(local_path, "wb") as f:
            f.write(image_data)
        
        print(f"✅ Emergency image saved locally: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"❌ Error saving emergency image locally: {e}")
        return None

# Backward compatibility alias
save_emergency_image_to_gcs = save_emergency_image

def send_emergency_email(report_data: dict, image_path: Optional[str] = None) -> bool:
    """Send emergency report email with image attachment"""
    try:
        if not EMAIL_PASSWORD:
            print("⚠️ Email password not set. Cannot send emergency alert.")
            return False
            
        print(f"📧 Sending emergency alert email for report {report_data['report_id']}...")
        
        # Create email message
        msg = EmailMessage()
        msg["Subject"] = f"🚨 EMERGENCY REPORT - {report_data['classification'].upper()} - {report_data['location']}"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        
        # Build email body
        body_lines = [
            f"🚨 EMERGENCY REPORT ALERT 🚨",
            f"",
            f"📍 Location: {report_data['location']}",
            f"🆔 Report ID: {report_data['report_id']}",
            f"🏷️ AI Classification: {report_data['classification'].upper()}",
            f"⏰ Reported At: {report_data['timestamp']}",
            f"",
            f"📝 EMERGENCY DESCRIPTION:",
            f"{report_data['message']}",
            f"",
            f"📞 Contact: {report_data.get('contact', 'Not provided')}",
            f"",
            f"🤖 AI ANALYSIS DETAILS:",
            f"├── Category: {get_category_description(report_data['classification'])}",
            f"├── Severity: {report_data.get('severity', 'UNKNOWN')}",
            f"├── Reasoning: {report_data.get('reasoning', 'No reasoning provided')}",
            f"├── Recommended Action: {report_data.get('recommended_action', 'Monitor situation')}",
            f"└── Priority: {get_priority_level(report_data['classification'])}",
            f"",
        ]
        
        if image_path:
            body_lines.extend([
                f"📷 IMAGE ATTACHMENT:",
                f"Emergency photo has been attached to this email.",
                f"Local Path: {image_path}",
                f"",
            ])
        
        body_lines.extend([
            f"⚠️ IMMEDIATE ACTION REQUIRED!",
            f"",
            f"This emergency report has been automatically classified by our AI system.",
            f"Please verify the situation and dispatch appropriate emergency response.",
            f"",
            f"This alert was generated by Gemma Kavach Emergency System"
        ])
        
        msg.set_content("\n".join(body_lines))
        
        # Attach image if available
        if image_path and report_data.get('image_data'):
            try:
                image_data = report_data['image_data']
                classification = report_data['classification']
                filename = f"emergency_{classification}_{report_data['report_id']}.jpg"
                
                msg.add_attachment(
                    image_data,
                    maintype="image",
                    subtype="jpeg",
                    filename=filename
                )
                print(f"✅ Image attached to email: {filename}")
            except Exception as e:
                print(f"⚠️ Failed to attach image: {e}")
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
            
        print(f"✅ Emergency alert email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send emergency email: {e}")
        return False

def get_category_description(classification: str) -> str:
    """Get human-readable description for classification"""
    descriptions = {
        'child_lost': 'Missing Child - High Priority Search Required',
        'crowd_panic': 'Crowd Control Emergency - Panic Situation Detected',
        'lost_item': 'Lost Property - Standard Recovery Procedure',
        'medical_help': 'Medical Emergency - Immediate Medical Attention Required',
        'need_interpreter': 'Language Assistance - Interpreter Support Needed',
        'small_fire': 'Fire Emergency - Fire Safety Response Required'
    }
    return descriptions.get(classification, f'Unknown Classification: {classification}')

def get_priority_level(classification: str) -> str:
    """Get priority level for classification"""
    priority_levels = {
        'child_lost': 'CRITICAL',
        'crowd_panic': 'CRITICAL', 
        'medical_help': 'HIGH',
        'small_fire': 'HIGH',
        'need_interpreter': 'MEDIUM',
        'lost_item': 'LOW'
    }
    return priority_levels.get(classification, 'MEDIUM')

def generate_report_id() -> str:
    """Generate unique emergency report ID"""
    timestamp = int(time.time())
    random_part = str(uuid.uuid4())[:8].upper()
    return f"EMG-{timestamp}-{random_part}"

def get_current_timestamp() -> str:
    """Get current timestamp in readable format"""
    return time.strftime("%Y-%m-%d %H:%M:%S")