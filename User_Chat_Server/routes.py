# routes.py - Complete file with classification + emergency endpoints
import os
import sys
import requests
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import logging

# Add Vision Server to path to reuse the audit logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Gemma_Kavach_Vision_Server")))
from audit_logger import log_emergency_report, log_audit_trail

from utils import (
    classify_emergency, 
    get_model_info,
    save_emergency_image,
    send_emergency_email,
    generate_report_id,
    get_current_timestamp
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request/Response models for classification
class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    category: str
    severity: str = "LOW"
    reasoning: str = "No reasoning provided."
    action: str = "Monitor"

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    categories: list
    model_path: str
    device: str

# Classification endpoints
@router.post("/ask_class", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Classify emergency text into predefined categories
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = classify_emergency(request.text.strip())
        category = result.get("classification", "unknown")
        severity = result.get("severity", "LOW")
        reasoning = result.get("reasoning", "No reasoning provided.")
        action = result.get("action", "Monitor")
        
        logger.info(f"Classification: '{request.text}' -> '{category}'")
        
        return ClassificationResponse(
            category=category,
            severity=severity,
            reasoning=reasoning,
            action=action
        )
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

@router.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information and status"""
    try:
        # Query GemmaServer for actual model status
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                model_loaded = health_data.get("model_loaded", False)
            else:
                model_loaded = False
        except:
            model_loaded = False
        
        info = get_model_info()
        info["model_loaded"] = model_loaded
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

@router.post("/batch_classify")
async def batch_classify(requests: list[ClassificationRequest]):
    """Classify multiple texts at once"""
    try:
        results = []
        for req in requests:
            if req.text and req.text.strip():
                result = classify_emergency(req.text.strip())
                category = result.get("classification", "unknown")
                results.append({
                    "text": req.text,
                    "category": category,
                    "severity": result.get("severity", "LOW"),
                    "reasoning": result.get("reasoning", ""),
                    "action": result.get("action", "")
                })
            else:
                results.append({
                    "text": req.text,
                    "category": "error",
                    "error": "empty_text"
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail="Batch classification failed")

@router.get("/debug")
async def debug_info():
    """Debug endpoint with actual GemmaServer status"""
    try:
        # Query GemmaServer for actual status
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            model_loaded = health_data.get("model_loaded", False)
            gemma_status = "connected"
        else:
            model_loaded = False
            gemma_status = f"error: {health_response.status_code}"
    except Exception as e:
        model_loaded = False
        gemma_status = f"unreachable: {str(e)}"
    
    return {
        "model_loaded": model_loaded,
        "tokenizer_loaded": model_loaded,
        "model_type": "Delegated to GemmaServer",
        "tokenizer_type": "Delegated to GemmaServer",
        "model_device": "Delegated to GemmaServer",
        "gemma_server_status": gemma_status,
        "categories": get_model_info()["categories"]
    }

# Emergency Report endpoints
@router.post("/emergency/report")
async def submit_emergency_report(
    background_tasks: BackgroundTasks,
    location: str = Form(...),
    message: str = Form(...),
    classification: str = Form(...), # We accept this but verify it server-side
    contact: str = Form(None),
    clientReportId: str = Form(None, alias="reportId"),
    image: UploadFile = File(...)
):
    """
    Submit emergency report with image and send email alert
    """
    try:
        # Generate immutable report ID server-side
        reportId = generate_report_id()
        print(f"📋 Processing emergency report: {reportId} (Client ID: {clientReportId})")
        print(f"📍 Location: {location}")
        
        # 🚨 CRITICAL FIX: Do not trust the client's classification!
        # Re-classify the emergency text server-side to prevent spoofing/errors
        print(f"🧠 Verifying classification server-side for message: '{message}'")
        verification_result = classify_emergency(message)
        verified_classification = verification_result.get("classification", classification)
        
        if verified_classification == "error":
            print("❌ Classification failed due to GemmaServer unavailability")
            raise HTTPException(status_code=503, detail="AI Classification service unavailable. Please route to human review queue.")
            
        severity = verification_result.get("severity", "UNKNOWN")
        reasoning = verification_result.get("reasoning", "No reasoning generated.")
        recommended_action = verification_result.get("action", "Monitor situation.")
        
        print(f"🏷️ Final Classification: {verified_classification} (Client suggested: {classification})")
        
        # Read image data
        image_data = await image.read()
        
        # Save image locally
        image_local_path = save_emergency_image(
            reportId, 
            image_data, 
            image.filename or "emergency.jpg"
        )
        
        # Prepare report data
        report_data = {
            'report_id': reportId,
            'location': location,
            'message': message,
            'classification': verified_classification,
            'severity': severity,
            'reasoning': reasoning,
            'recommended_action': recommended_action,
            'contact': contact,
            'timestamp': get_current_timestamp(),
            'image_data': image_data,
            'image_path': image_local_path
        }
        
        # Log to the database (emergency_reports table)
        log_emergency_report(
            report_id=reportId,
            message=message,
            classification=verified_classification,
            severity=severity,
            reasoning=reasoning,
            recommended_action=recommended_action,
            location=location,
            contact=contact,
            image_path=image_local_path
        )
        
        # Log to audit trail
        log_audit_trail(
            report_id=reportId,
            event_type="emergency_report_submitted",
            severity=severity,
            reasoning=reasoning,
            action_taken=recommended_action,
            metadata={"classification": verified_classification, "location": location}
        )
        
        # Send email in background
        background_tasks.add_task(
            send_emergency_email, 
            report_data, 
            image_local_path
        )
        
        print(f"✅ Emergency report processed and logged to DB: {reportId}")
        
        return {
            "status": "success",
            "report_id": reportId,
            "message": "Emergency report submitted, verified, logged, and alert sent",
            "image_saved": image_local_path is not None,
            "timestamp": report_data['timestamp'],
            "verified_classification": verified_classification
        }
        
    except Exception as e:
        print(f"❌ Error processing emergency report: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process emergency report: {str(e)}"
        )

@router.get("/emergency/status/{report_id}")
async def get_emergency_status(report_id: str):
    """Get status of emergency report (for future tracking)"""
    return {
        "report_id": report_id,
        "status": "processed",
        "message": "Emergency report has been submitted, verified, and alerts sent",
        "timestamp": get_current_timestamp()
    }