import os
from typing import Dict, Any
import aiohttp
from fastapi import APIRouter, UploadFile, File, HTTPException
from .config import settings

router = APIRouter()

class AzureServices:
    """Azure Cognitive Services integration with fallback mode"""

    def __init__(self):
        self.vision_key = settings.AZURE_COMPUTER_VISION_KEY
        self.vision_endpoint = settings.AZURE_COMPUTER_VISION_ENDPOINT
        self.speech_key = settings.AZURE_SPEECH_KEY
        self.speech_region = settings.AZURE_SPEECH_REGION

    async def analyze_image(self, file_path: str) -> Dict[str, Any]:
        """Analyze image using Azure Vision, or fallback if not configured"""
        if not self.vision_key or not self.vision_endpoint:
            # 🔹 Fallback mode (no Azure)
            return {
                "success": True,
                "score": 0.75,
                "analysis": {"description": "Demo fallback image analysis"},
                "service": "azure-vision-fallback"
            }

        try:
            with open(file_path, "rb") as f:
                image_data = f.read()

            headers = {
                "Ocp-Apim-Subscription-Key": self.vision_key,
                "Content-Type": "application/octet-stream"
            }
            url = f"{self.vision_endpoint}/vision/v3.2/analyze"
            params = {"visualFeatures": "Description,Tags,Objects,Faces,ImageType,Color,Adult"}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, params=params, data=image_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "score": 0.8, "analysis": result, "service": "azure-vision"}
                    else:
                        return {"success": False, "error": f"Azure Vision API error: {response.status}"}
        except Exception as e:
            return {"success": False, "error": f"Azure Vision failed: {str(e)}"}

    async def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio using Azure Speech, or fallback if not configured"""
        if not self.speech_key or not self.speech_region:
            # 🔹 Fallback mode (no Azure)
            return {
                "success": True,
                "score": 0.6,
                "analysis": {"transcript": "This is a demo fallback transcript."},
                "service": "azure-speech-fallback"
            }

        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()

            headers = {
                "Ocp-Apim-Subscription-Key": self.speech_key,
                "Content-Type": "audio/wav"
            }
            url = f"https://{self.speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=audio_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "score": 0.85, "analysis": result, "service": "azure-speech"}
                    else:
                        return {"success": False, "error": f"Azure Speech API error: {response.status}"}
        except Exception as e:
            return {"success": False, "error": f"Azure Speech failed: {str(e)}"}


# ✅ FastAPI endpoints
azure_service = AzureServices()

@router.post("/image")
async def analyze_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        result = await azure_service.analyze_image(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        result = await azure_service.analyze_audio(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
