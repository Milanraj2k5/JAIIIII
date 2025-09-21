import os
from typing import Dict, Any, Optional
import asyncio
import aiohttp
import base64
from .config import settings

class AzureServices:
    """Azure Cognitive Services integration for media analysis"""
    
    def __init__(self):
        self.vision_key = settings.AZURE_VISION_KEY
        self.vision_endpoint = settings.AZURE_VISION_ENDPOINT
        self.speech_key = settings.AZURE_SPEECH_KEY
        self.speech_endpoint = settings.AZURE_SPEECH_ENDPOINT
    
    async def analyze_image(self, file_path: str) -> Dict[str, Any]:
        """Analyze image using Azure Computer Vision"""
        if not self.vision_key or not self.vision_endpoint:
            raise Exception("Azure Vision credentials not configured")
        
        try:
            # Read and encode image
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Call Azure Computer Vision API
            headers = {
                'Ocp-Apim-Subscription-Key': self.vision_key,
                'Content-Type': 'application/octet-stream'
            }
            
            url = f"{self.vision_endpoint}/vision/v3.2/analyze"
            params = {
                'visualFeatures': 'Description,Tags,Objects,Faces,ImageType,Color,Adult'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, params=params, data=image_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract relevant information for trust scoring
                        confidence_score = 0.8  # Default confidence
                        
                        # Check for suspicious elements
                        if 'adult' in result and result['adult']['isAdultContent']:
                            confidence_score -= 0.2
                        
                        if 'description' in result and 'captions' in result['description']:
                            captions = result['description']['captions']
                            if captions and any('fake' in caption['text'].lower() for caption in captions):
                                confidence_score -= 0.3
                        
                        return {
                            "score": max(0.0, min(1.0, confidence_score)),
                            "analysis": result,
                            "service": "azure-vision",
                            "confidence": confidence_score
                        }
                    else:
                        raise Exception(f"Azure API error: {response.status}")
        
        except Exception as e:
            print(f"Azure Vision analysis failed: {e}")
            raise e
    
    async def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio using Azure Speech Services"""
        if not self.speech_key or not self.speech_endpoint:
            raise Exception("Azure Speech credentials not configured")
        
        try:
            # Read audio file
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Call Azure Speech API for speaker recognition
            headers = {
                'Ocp-Apim-Subscription-Key': self.speech_key,
                'Content-Type': 'audio/wav'
            }
            
            url = f"{self.speech_endpoint}/speech/recognition/conversation/cognitiveservices/v1"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=audio_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract confidence for trust scoring
                        confidence = result.get('RecognitionStatus', 'Success')
                        if confidence == 'Success':
                            confidence_score = 0.85
                        else:
                            confidence_score = 0.3
                        
                        return {
                            "score": confidence_score,
                            "analysis": result,
                            "service": "azure-speech",
                            "confidence": confidence_score
                        }
                    else:
                        raise Exception(f"Azure Speech API error: {response.status}")
        
        except Exception as e:
            print(f"Azure Speech analysis failed: {e}")
            raise e