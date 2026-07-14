# routes.py - API endpoints with dual crowd analysis, reasoning output, and audit trail
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import aiohttp
import asyncio
from utils import (
    save_session, 
    load_session,
    save_flagged_image,
    should_send_alert,
    send_alert_email,
    generate_session_id, 
    get_current_timestamp,
    get_verdict,
    check_db_connection,
    calculate_risk_score,
)
from reasoning_engine import generate_vision_reasoning
from audit_logger import (
    log_frame_analysis,
    log_alert,
    log_audit_trail,
    get_full_audit_trail,
    get_alerts,
)

# Create router
router = APIRouter()

# Gemma server configuration — LOCAL inference, no cloud dependency
GEMMA_API_URL = "http://localhost:8000/ask_image"

# Improved analysis prompts
CROWD_DENSITY_PROMPT = (
    "Analyze this image and determine the crowd density level. "
    "Look at how tightly packed people are together:\n"
    "- Low: People have plenty of space, sparse crowd\n"
    "- Medium: Moderate crowding, some personal space\n"
    "- High: Very dense, people tightly packed with minimal space\n\n"
    "Respond with only one word: 'Low', 'Medium', or 'High'."
)

CROWD_MOTION_PROMPT = (
    "Analyze this image for signs of panic behavior or chaotic crowd movement. "
    "Look for:\n"
    "- People running, pushing, or shoving\n"
    "- Panic expressions or body language\n"
    "- Chaotic, disorganized movement patterns\n"
    "- People falling or being trampled\n"
    "- Any signs of fear or distress\n\n"
    "Respond with only one word: 'Calm' if no panic signs, or 'Chaotic' if panic behavior detected."
)

# Enhanced response models
class FrameAnalysisResponse(BaseModel):
    session_id: str
    frame_number: int
    crowd_density: str  # "Low", "Medium", "High"
    crowd_motion: str   # "Calm", "Chaotic"
    risk_detected: bool
    risk_level: str     # "SAFE", "MODERATE", "HIGH", "CRITICAL"
    updated_risk_score: float
    frames_analyzed: int
    frames_flagged: int
    timestamp: str
    analysis_details: Dict[str, Any]
    reasoning: Optional[Dict[str, Any]] = None  # Populated on flagged frames

class CreateSessionRequest(BaseModel):
    location: str = "Mela Zone B"
    operator_name: Optional[str] = "Security Team"

class SessionResponse(BaseModel):
    session_id: str
    status: str
    location: str
    timestamp: str
    message: str

class SessionStatusResponse(BaseModel):
    session_id: str
    location: str
    operator_name: str
    status: str
    created_at: str
    last_analysis: Optional[str]
    frames_analyzed: int
    frames_flagged: int
    risk_score: float
    verdict: str
    analysis_breakdown: Optional[Dict[str, Any]]

async def analyze_frame_dual(image_content: bytes, filename: str) -> Dict[str, str]:
    """
    Perform dual analysis on a single frame using async calls
    Returns: {"density": "Low/Medium/High", "motion": "Calm/Chaotic"}
    """
    async def call_gemma_api(prompt: str, session: aiohttp.ClientSession) -> str:
        """Make a single API call to Gemma"""
        try:
            data = aiohttp.FormData()
            data.add_field('image', image_content, filename=filename, content_type='image/jpeg')
            data.add_field('prompt', prompt)
            
            async with session.post(GEMMA_API_URL, data=data) as response:
                if response.status != 200:
                    raise Exception(f"Gemma API error: {response.status}")
                
                result = await response.json()
                return result.get("text", "").strip()
                
        except Exception as e:
            print(f"❌ Gemma API call failed: {e}")
            return "Unknown"
    
    # Make both API calls concurrently
    async with aiohttp.ClientSession() as session:
        density_task = call_gemma_api(CROWD_DENSITY_PROMPT, session)
        motion_task = call_gemma_api(CROWD_MOTION_PROMPT, session)
        
        # Wait for both results
        density_result, motion_result = await asyncio.gather(density_task, motion_task)
        
        # Clean and validate results
        density = density_result.title() if density_result.lower() in ['low', 'medium', 'high'] else 'Unknown'
        motion = motion_result.title() if motion_result.lower() in ['calm', 'chaotic'] else 'Unknown'
        
        return {
            "density": density,
            "motion": motion
        }

def determine_risk_level(density: str, motion: str) -> tuple[bool, str]:
    """
    Determine risk based on crowd density and motion
    Returns: (risk_detected: bool, risk_level: str)
    """
    # Risk matrix logic
    if motion == "Chaotic":
        if density == "High":
            return True, "CRITICAL"  # High density + chaos = maximum danger
        elif density == "Medium":
            return True, "HIGH"      # Medium density + chaos = high danger
        else:  # Low density
            return True, "MODERATE"  # Even low density chaos is concerning
    
    elif density == "High":
        if motion == "Calm":
            return True, "MODERATE"  # High density but calm = watch closely
        else:  # Unknown motion
            return True, "MODERATE"  # High density with unknown motion
    
    elif density == "Medium":
        return False, "SAFE"         # Medium density + calm = safe
    
    else:  # Low density or Unknown
        return False, "SAFE"         # Low density = safe

@router.post("/session/create", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new crowd monitoring session"""
    try:
        session_id = generate_session_id()
        timestamp = get_current_timestamp()
        
        session_data = {
            "session_id": session_id,
            "location": request.location,
            "operator_name": request.operator_name,
            "status": "created",
            "created_at": timestamp,
            "frames_analyzed": 0,
            "frames_flagged": 0,
            "risk_score": 0.0,
            "analysis_breakdown": {
                "density_stats": {"Low": 0, "Medium": 0, "High": 0, "Unknown": 0},
                "motion_stats": {"Calm": 0, "Chaotic": 0, "Unknown": 0},
                "risk_levels": {"SAFE": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
            }
        }
        
        success = save_session(session_id, session_data)
        if not success:
            raise Exception("Failed to save session to database")
        
        # Log session creation to audit trail
        log_audit_trail(
            session_id=session_id,
            event_type="session_created",
            severity="INFO",
            reasoning=f"New monitoring session created for {request.location}",
            action_taken="Session initialized",
            metadata={"location": request.location, "operator": request.operator_name}
        )
        
        print(f"✅ Created session: {session_id}")
        print(f"📍 Location: {request.location}")
        print(f"👤 Operator: {request.operator_name}")
        
        return SessionResponse(
            session_id=session_id,
            status="created",
            location=request.location,
            timestamp=timestamp,
            message="Session created successfully with dual-analysis capability."
        )
        
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.post("/session/{session_id}/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(session_id: str, background_tasks: BackgroundTasks, frame: UploadFile = File(...)):
    """
    Analyze a single frame using dual crowd analysis (density + motion).
    Returns reasoning and evidence for flagged frames.
    """
    try:
        # Load existing session
        session_data = load_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        print(f"🧠 Analyzing frame for session: {session_id}")
        
        # Prepare image for analysis
        image_content = await frame.read()
        frame_number = session_data["frames_analyzed"] + 1
        
        # Perform dual analysis (async)
        analysis_start = asyncio.get_event_loop().time()
        analysis_results = await analyze_frame_dual(image_content, frame.filename)
        analysis_time = round(asyncio.get_event_loop().time() - analysis_start, 2)
        
        density = analysis_results["density"]
        motion = analysis_results["motion"]
        
        # Determine risk
        risk_detected, risk_level = determine_risk_level(density, motion)
        
        # Update session statistics
        session_data["frames_analyzed"] = frame_number
        
        # Update breakdown stats
        breakdown = session_data.get("analysis_breakdown", {
            "density_stats": {"Low": 0, "Medium": 0, "High": 0, "Unknown": 0},
            "motion_stats": {"Calm": 0, "Chaotic": 0, "Unknown": 0},
            "risk_levels": {"SAFE": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
        })
        
        breakdown["density_stats"][density] += 1
        breakdown["motion_stats"][motion] += 1
        breakdown["risk_levels"][risk_level] += 1
        session_data["analysis_breakdown"] = breakdown
        
        # Reasoning data — populated only for flagged frames
        reasoning_data = None
        image_path = None
        
        # Handle flagged frames
        if risk_detected:
            session_data["frames_flagged"] += 1
            
            # Save flagged image locally
            image_path = save_flagged_image(session_id, frame_number, image_content)
            
            # Generate reasoning for this risk
            reasoning = await generate_vision_reasoning(density, motion, risk_level)
            
            # Build structured reasoning response
            reasoning_data = {
                "risk_level": risk_level,
                "reasoning": reasoning.get("reasoning", f"Risk detected: {density} density with {motion} motion."),
                "action": reasoning.get("action", "Monitor and investigate."),
                "evidence": {
                    "density": density,
                    "motion": motion,
                    "frame_number": frame_number,
                    "analysis_time_seconds": analysis_time,
                },
                "timestamp": get_current_timestamp(),
            }
            
            # Log alert to the alerts table
            log_alert(
                session_id=session_id,
                risk_level=risk_level,
                reasoning=reasoning_data["reasoning"],
                evidence=reasoning_data["evidence"],
                frame_refs=str(frame_number),
            )
            
            # Log audit trail for this flagged frame
            log_audit_trail(
                session_id=session_id,
                event_type="vision_alert",
                severity=risk_level,
                reasoning=reasoning_data["reasoning"],
                action_taken=reasoning_data["action"],
                metadata={"frame_number": frame_number, "density": density, "motion": motion}
            )
            
            # Add detailed flagged frame info
            if "flagged_frames" not in session_data:
                session_data["flagged_frames"] = []
                
            session_data["flagged_frames"].append({
                "frame_number": frame_number,
                "local_path": image_path,
                "timestamp": get_current_timestamp(),
                "crowd_density": density,
                "crowd_motion": motion,
                "risk_level": risk_level,
                "analysis_time_seconds": analysis_time,
                "reasoning": reasoning_data
            })
            
        # Log frame analysis to SQLite
        log_frame_analysis(session_id, frame_number, density, motion, risk_level, risk_detected, image_path)
        
        # Calculate enhanced risk score
        risk_score = calculate_risk_score(session_data)
        session_data["risk_score"] = risk_score
        session_data["last_analysis"] = get_current_timestamp()
        
        # Save updated session
        save_success = save_session(session_id, session_data)
        if not save_success:
            print("⚠️ Failed to save session update")
        
        # Check for alert (background task)
        if should_send_alert(session_data):
            print(f"🚨 Alert criteria met! Sending email in background...")
            session_data["email_sent"] = True
            save_session(session_id, session_data)
            background_tasks.add_task(send_alert_email, session_id, session_data)
        
        # Log analysis results
        risk_emoji = "🔴" if risk_level == "CRITICAL" else "🟠" if risk_level == "HIGH" else "🟡" if risk_level == "MODERATE" else "🟢"
        print(f"📊 Frame {frame_number}: {risk_emoji} {risk_level}")
        print(f"   Density: {density} | Motion: {motion}")
        print(f"   Risk Score: {risk_score}% | Time: {analysis_time}s")
        
        return FrameAnalysisResponse(
            session_id=session_id,
            frame_number=frame_number,
            crowd_density=density,
            crowd_motion=motion,
            risk_detected=risk_detected,
            risk_level=risk_level,
            updated_risk_score=risk_score,
            frames_analyzed=frame_number,
            frames_flagged=session_data["frames_flagged"],
            timestamp=get_current_timestamp(),
            analysis_details={
                "analysis_time_seconds": analysis_time,
                "density_breakdown": breakdown["density_stats"],
                "motion_breakdown": breakdown["motion_stats"],
                "risk_level_breakdown": breakdown["risk_levels"]
            },
            reasoning=reasoning_data,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze frame: {str(e)}")

@router.get("/session/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get current status and detailed analytics of a monitoring session"""
    try:
        session_data = load_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        risk_score = session_data.get("risk_score", 0.0)
        verdict = get_verdict(risk_score)
        
        print(f"📊 Session {session_id} status: {verdict} ({risk_score}%)")
        
        return SessionStatusResponse(
            session_id=session_id,
            location=session_data["location"],
            operator_name=session_data["operator_name"],
            status=session_data["status"],
            created_at=session_data["created_at"],
            last_analysis=session_data.get("last_analysis"),
            frames_analyzed=session_data["frames_analyzed"],
            frames_flagged=session_data["frames_flagged"],
            risk_score=risk_score,
            verdict=verdict,
            analysis_breakdown=session_data.get("analysis_breakdown")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")


# ---------------------------------------------------------------------------
# Audit trail API — read-only endpoints for forensic review
# ---------------------------------------------------------------------------

@router.get("/audit/{session_id}")
async def get_session_audit_trail(session_id: str, limit: int = 100):
    """Get the full audit trail for a specific session (read-only, forensic-grade)"""
    trail = get_full_audit_trail(session_id=session_id, limit=limit)
    alerts_list = get_alerts(session_id=session_id, limit=limit)
    return {
        "session_id": session_id,
        "audit_trail": trail,
        "alerts": alerts_list,
        "total_entries": len(trail),
        "total_alerts": len(alerts_list),
    }

@router.get("/audit")
async def get_recent_audit_trail(limit: int = 50):
    """Get recent audit trail entries across all sessions (read-only)"""
    trail = get_full_audit_trail(limit=limit)
    alerts_list = get_alerts(limit=limit)
    return {
        "audit_trail": trail,
        "alerts": alerts_list,
        "total_entries": len(trail),
        "total_alerts": len(alerts_list),
    }


@router.get("/monitoring/status")
async def monitoring_status():
    """Check if monitoring service is available"""
    db_info = check_db_connection()
    
    return {
        "service": "monitoring",
        "status": "available",
        "analysis_type": "dual_crowd_analysis",
        "features": [
            "crowd_density_detection",
            "panic_behavior_detection",
            "async_api_calls",
            "reasoning_per_alert",
            "audit_trail",
        ],
        "database": db_info,
        "gemma_api": GEMMA_API_URL,
        "endpoints": [
            "/session/create",
            "/session/{id}/frame",
            "/session/{id}",
            "/audit/{session_id}",
            "/audit",
        ]
    }