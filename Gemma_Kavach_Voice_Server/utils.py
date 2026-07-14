# Enhanced utils.py with Google Text-to-Speech

import requests
import base64
import pandas as pd
import os
from google.cloud import texttospeech
import tempfile
import json

# Configuration
SERVER_URL = "http://localhost:8000/"
TEST_MODE = False
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TEXT_TO_SPEECH")

def transcribe_audio_from_bytes(audio_bytes: bytes, prompt="Transcribe this audio"):
    """Get transcription from audio bytes with enhanced error handling and fallback"""
    try:
        # Convert audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Try the main transcription endpoint first
        data = {"data": audio_base64, "prompt": "Transcribe this in Hindi"}
        
        print(f"📤 Sending audio data ({len(audio_bytes)} bytes) to RunPod server...")
        
        # Try multiple times with different configurations
        for attempt in range(3):
            try:
                response = requests.post(f"{SERVER_URL}ask", json=data, timeout=90)
                
                if response.status_code == 200:
                    result = response.json()
                    if "text" in result and result["text"].strip():
                        transcribed_text = result["text"].strip()
                        print(f"✅ Transcription successful (attempt {attempt + 1}): {transcribed_text}")
                        return transcribed_text
                    else:
                        print(f"⚠️ Empty transcription result (attempt {attempt + 1})")
                        if attempt == 2:  # Last attempt
                            return "No speech detected"
                else:
                    print(f"❌ Transcription API error (attempt {attempt + 1}): {response.status_code}")
                    if attempt == 2:  # Last attempt
                        return "TRANSCRIPTION_FAILED"
                        
            except requests.exceptions.Timeout:
                print(f"⏱️ Request timeout (attempt {attempt + 1})")
                if attempt == 2:
                    return "TRANSCRIPTION_FAILED"
            except requests.exceptions.ConnectionError:
                print(f"🔌 Connection error (attempt {attempt + 1})")
                if attempt == 2:
                    return "TRANSCRIPTION_FAILED"
            except Exception as e:
                print(f"❌ Transcription error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return "TRANSCRIPTION_FAILED"
                    
            # Wait before retrying
            if attempt < 2:
                print(f"🔄 Retrying in {2 ** attempt} seconds...")
                import time
                time.sleep(2 ** attempt)
        
        return "TRANSCRIPTION_FAILED"
            
    except Exception as e:
        print(f"❌ Critical transcription error: {e}")
        return "TRANSCRIPTION_FAILED"

def extract_zone_info(text):
    """Extract zone information from transcribed text"""
    prompt = f"""Extract the zone information from this text. Return ONLY the zone letter/number. No other text.

Example:
Input: "जमा जी ज़ोन सी की सिक्योरिटी अपडेट दीजिए प्लीज।"
Output: C

Input: "ज़ोन बी की अपडेट चाहिए"
Output: B

Input: "{text}"
Output:"""

    data = {"prompt": prompt, "max_tokens": 10, "processing_mode": "force_off"}
    
    response = requests.post(f"{SERVER_URL}/generate", json=data, timeout=90)
    
    if response.status_code == 200:
        zone = response.json()["text"].strip()
        # Only accept single English letters/numbers
        if zone and len(zone) <= 2 and zone.isalnum():
            return {"zone": zone}
    
    return None

def get_zone_update(zone_name):
    """Get latest update for the zone from SQLite database"""
    try:
        import sqlite3
        import json
        
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "kavach.db"))
        if not os.path.exists(db_path):
            return "❌ Database file 'kavach.db' not found. Please run the vision server first."
            
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the most recent session for the requested zone
        cursor.execute("SELECT * FROM sessions WHERE location = ? ORDER BY last_analysis DESC LIMIT 1", (zone_name,))
        latest_session = cursor.fetchone()
        
        if not latest_session:
            cursor.execute("SELECT DISTINCT location FROM sessions")
            locations = [row['location'] for row in cursor.fetchall()]
            conn.close()
            return f"❌ No data found for {zone_name}. Available zones: {locations}"
            
        session_id = latest_session['session_id']
        risk_score = latest_session['risk_score']
        frames_analyzed = latest_session['frames_analyzed']
        frames_flagged = latest_session['frames_flagged']
        last_update = latest_session['last_analysis']
        operator_name = latest_session['operator_name'] or 'Security Team'
        
        # Parse breakdown
        breakdown_str = latest_session['analysis_breakdown']
        breakdown = json.loads(breakdown_str) if breakdown_str else {}
        
        density_high = breakdown.get('density_stats', {}).get('High', 0)
        motion_chaotic = breakdown.get('motion_stats', {}).get('Chaotic', 0)
        risk_critical = breakdown.get('risk_levels', {}).get('CRITICAL', 0)
        
        conn.close()
        
        # Calculate additional metrics
        flagging_rate = (frames_flagged / frames_analyzed * 100) if frames_analyzed > 0 else 0
        
        # Determine risk status based on score
        if risk_score <= 15:
            status = "🟢 SAFE"
            status_msg = "operating normally"
        elif risk_score <= 40:
            status = "🟡 WATCH"
            status_msg = "under routine monitoring"
        elif risk_score <= 70:
            status = "🟠 ALERT"
            status_msg = "requires attention"
        else:
            status = "🔴 CRITICAL"
            status_msg = "IMMEDIATE ACTION REQUIRED"
        
        # Create comprehensive update message
        message = f"""
🛡️ {zone_name} Security Update

📊 Current Status: {status}
📈 Risk Score: {risk_score}%
👥 Frames Analyzed: {frames_analyzed}
⚠️ Frames Flagged: {frames_flagged} ({flagging_rate:.1f}%)
🕒 Last Updated: {last_update}
👤 Operator: {operator_name}
🆔 Session: {session_id}

📋 Analysis Details:
• High Density Events: {density_high}
• Chaotic Motion Events: {motion_chaotic}  
• Critical Risk Events: {risk_critical}

Status: Zone is currently {status_msg}.
"""
        
        return message.strip()
        
    except Exception as e:
        return f"❌ Error retrieving update for {zone_name}: {str(e)}"

def generate_hindi_message(zone_update_data):
    """Generate user-friendly Hindi message using LLM"""
    try:
        prompt = f"""Convert this security update data into a natural, user-friendly Hindi message for voice response. Make it conversational and easy to understand.

Data: {zone_update_data}

Instructions:
- Convert to natural Hindi 
- Make it sound like a security officer giving an update
- Keep it concise but informative
- Include key status and numbers
- Sound professional but friendly
- Do NOT say "जी हाँ, ज़ोन बी" - just give the update directly
- Start with the current status

Example style: "स्थिति सामान्य है। रिस्क स्कोर 15% है और कोई खतरा नहीं है।"

Hindi Message:"""

        data = {"prompt": prompt, "max_tokens": 200}
        
        response = requests.post(f"{SERVER_URL}/generate", json=data, timeout=90)
        
        if response.status_code == 200:
            hindi_message = response.json()["text"].strip()
            return hindi_message
        else:
            return "अपडेट प्राप्त करने में कुछ समस्या है। कृपया दोबारा कोशिश करें।"
            
    except Exception as e:
        print(f"❌ Error generating Hindi message: {e}")
        return "तकनीकी समस्या के कारण अपडेट नहीं मिल सका।"

def generate_speech_audio(text: str, language_code: str = "hi-IN") -> str:
    """
    Convert text to speech using Google Text-to-Speech API
    Returns base64 encoded audio data
    """
    try:
        if not GOOGLE_TTS_API_KEY:
            print("⚠️ Google TTS API key not found, skipping audio generation")
            return None
            
        print(f"🎵 Generating speech for text: {text[:50]}...")
        
        # Prepare the request payload
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
        
        # Configure voice settings for Hindi
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": "hi-IN-Wavenet-A",  # Female Hindi voice
                "ssmlGender": "FEMALE"
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 0.9,  # Slightly slower for clarity
                "pitch": 0.0,
                "volumeGainDb": 0.0
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 200:
            result = response.json()
            audio_content = result.get("audioContent")
            
            if audio_content:
                print("✅ Google TTS audio generated successfully")
                return audio_content  # Already base64 encoded
            else:
                print("❌ No audio content in response")
                return None
                
        else:
            print(f"❌ Google TTS API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        return None

def generate_speech_with_fallback(text: str) -> str:
    """
    Generate speech with fallback options
    """
    # Try Google TTS first
    audio_content = generate_speech_audio(text, "hi-IN")
    
    if audio_content:
        return audio_content
    
    # Try English if Hindi fails
    print("🔄 Trying English voice as fallback...")
    audio_content = generate_speech_audio(text, "en-IN")
    
    if audio_content:
        return audio_content
    
    print("❌ All TTS options failed")
    return None