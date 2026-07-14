# routes.py - Complete file with classification + emergency endpoints
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from utils import (
    classify_emergency, 
    get_model_info,
    save_emergency_image_to_gcs,
    send_emergency_email,
    generate_report_id,
    get_current_timestamp
)
import logging

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

# Classification endpoints (your existing working endpoints)
@router.post("/ask_class", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Classify emergency text into predefined categories
    
    Categories: child_lost, crowd_panic, lost_item, medical_help, need_interpreter, small_fire
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Classify
        result = classify_emergency(request.text.strip())
        category = result.get("classification", "unknown")
        severity = result.get("severity", "LOW")
        reasoning = result.get("reasoning", "No reasoning provided.")
        action = result.get("action", "Monitor")
        
        # Log the request
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
        info = get_model_info()
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
    return {
        "model_loaded": True,
        "tokenizer_loaded": True,
        "model_type": "Delegated to GemmaServer",
        "tokenizer_type": "Delegated to GemmaServer",
        "model_device": "Delegated to GemmaServer",
        "categories": get_model_info()["categories"]
    }

# Emergency Report endpoints (new)
@router.post("/emergency/report")
async def submit_emergency_report(
    background_tasks: BackgroundTasks,
    location: str = Form(...),
    message: str = Form(...),
    classification: str = Form(...),
    contact: str = Form(None),
    reportId: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Submit emergency report with image and send email alert
    """
    try:
        print(f"📋 Processing emergency report: {reportId}")
        print(f"📍 Location: {location}")
        print(f"🏷️ Classification: {classification}")
        
        # Read image data
        image_data = await image.read()
        
        # Save image to GCS
        image_gcs_path = save_emergency_image_to_gcs(
            reportId, 
            image_data, 
            image.filename or "emergency.jpg"
        )
        
        # Prepare report data
        report_data = {
            'report_id': reportId,
            'location': location,
            'message': message,
            'classification': classification,
            'contact': contact,
            'timestamp': get_current_timestamp(),
            'image_data': image_data,
            'image_gcs_path': image_gcs_path
        }
        
        # Send email in background
        background_tasks.add_task(
            send_emergency_email, 
            report_data, 
            image_gcs_path
        )
        
        print(f"✅ Emergency report processed: {reportId}")
        
        return {
            "status": "success",
            "report_id": reportId,
            "message": "Emergency report submitted and alert sent",
            "image_saved": image_gcs_path is not None,
            "timestamp": report_data['timestamp']
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
        "message": "Emergency report has been submitted and alerts sent",
        "timestamp": get_current_timestamp()
    }