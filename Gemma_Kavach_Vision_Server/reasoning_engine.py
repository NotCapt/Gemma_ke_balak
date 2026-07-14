import aiohttp
import json
import logging

GEMMA_API_URL = "http://localhost:8000/classify"

async def generate_vision_reasoning(density: str, motion: str, risk_level: str) -> dict:
    """Generate structured reasoning for a detected crowd risk."""
    prompt = f"Crowd analysis detected {density} density and {motion} motion, leading to a {risk_level} risk level. Explain why this is dangerous and what actions should be taken. Provide a concise explanation."
    
    payload = {
        "text": prompt,
        "max_tokens": 150
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GEMMA_API_URL, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # We expect the /classify endpoint to return {category, severity, reasoning, action}
                    return result
                else:
                    return {
                        "category": "Crowd_Safety",
                        "severity": risk_level,
                        "reasoning": f"High risk due to {density} density and {motion} motion.",
                        "action": "Monitor and dispatch personnel if needed."
                    }
    except Exception as e:
        logging.error(f"Reasoning engine failed: {e}")
        return {
            "category": "Crowd_Safety",
            "severity": risk_level,
            "reasoning": f"System determined {risk_level} risk based on {density} density and {motion} motion.",
            "action": "Investigate immediately."
        }
