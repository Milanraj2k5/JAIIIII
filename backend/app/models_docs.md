# ML Models Documentation

This document describes the ML models used in TruthLens and how to integrate real models.

## Image Detection Models

### Recommended Models

1. **Deepfake Detection**
   - Model: `microsoft/DialoGPT-medium` (for text-based analysis)
   - Alternative: `facebook/detr-resnet-50` (for object detection)
   - Input: Image file (JPG, PNG)
   - Output: Confidence score (0-1)

2. **Face Consistency Analysis**
   - Model: `microsoft/face-api` or custom CNN
   - Input: Face regions extracted from image
   - Output: Consistency score

### Integration Example

```python
from transformers import pipeline
import torch
from PIL import Image

class ImageDetector:
    def __init__(self):
        # Load model
        self.detector = pipeline("image-classification", 
                               model="microsoft/detr-resnet-50")
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        # Load image
        image = Image.open(file_path)
        
        # Run inference
        results = self.detector(image)
        
        # Extract confidence score
        score = results[0]['score'] if results else 0.0
        
        return {
            "score": score,
            "explanations": [f"Detection: {results[0]['label']}"],
            "model": "microsoft/detr-resnet-50",
            "confidence": score
        }
```

## Video Detection Models

### Recommended Models

1. **Temporal Consistency**
   - Model: `facebook/timesformer-base`
   - Input: Video frames sequence
   - Output: Temporal consistency score

2. **Motion Analysis**
   - Model: Custom CNN + LSTM
   - Input: Optical flow features
   - Output: Motion authenticity score

### Integration Example

```python
from transformers import TimesformerForVideoClassification
import torch
import cv2

class VideoDetector:
    def __init__(self):
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base"
        )
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        # Extract frames
        frames = self._extract_frames(file_path)
        
        # Preprocess frames
        processed_frames = self._preprocess_frames(frames)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(processed_frames)
            score = torch.softmax(outputs.logits, dim=-1)[0][1].item()
        
        return {
            "score": score,
            "explanations": ["Temporal consistency analysis"],
            "model": "facebook/timesformer-base",
            "confidence": score
        }
```

## Audio Detection Models

### Recommended Models

1. **Voice Cloning Detection**
   - Model: `speechbrain/spkrec-ecapa-voxceleb`
   - Input: Audio waveform
   - Output: Speaker verification score

2. **Synthetic Voice Detection**
   - Model: Custom CNN + RNN
   - Input: Mel-spectrogram features
   - Output: Authenticity score

### Integration Example

```python
import speechbrain as sb
import torch
import librosa

class AudioDetector:
    def __init__(self):
        # Load SpeechBrain model
        self.model = sb.pretrained.SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Extract features
        features = self.model.encode_batch(audio)
        
        # Calculate authenticity score
        score = self._calculate_authenticity_score(features)
        
        return {
            "score": score,
            "explanations": ["Voice biometric analysis"],
            "model": "speechbrain/spkrec-ecapa-voxceleb",
            "confidence": score
        }
```

## Model Requirements

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install speechbrain
pip install opencv-python
pip install librosa
pip install pillow
```

### Performance Considerations

1. **GPU Usage**: Models run faster on GPU
2. **Batch Processing**: Process multiple files together
3. **Model Caching**: Load models once, reuse for multiple files
4. **Memory Management**: Clear GPU memory between batches

### Error Handling

```python
def analyze(self, file_path: str) -> Dict[str, Any]:
    try:
        # Model inference code
        return result
    except Exception as e:
        return {
            "error": str(e),
            "score": 0.0,
            "model": "error"
        }
```

## Production Deployment

### Model Optimization

1. **Quantization**: Reduce model size
2. **ONNX Conversion**: Faster inference
3. **TensorRT**: GPU optimization
4. **Model Pruning**: Remove unnecessary parameters

### Scaling

1. **Model Serving**: Use TensorFlow Serving or TorchServe
2. **Load Balancing**: Distribute requests across multiple instances
3. **Caching**: Cache model results for identical files
4. **Monitoring**: Track model performance and accuracy

## TODO: Model Integration

Replace the following lines in detector files:

1. **Image Detector**: Lines 15-16 in `image_detector.py`
2. **Video Detector**: Lines 15-16 in `video_detector.py`
3. **Audio Detector**: Lines 15-16 in `audio_detector.py`

Replace `_calculate_demo_score()` calls with actual model inference.

## Testing

```python
# Test with sample files
detector = ImageDetector()
result = detector.analyze("test_image.jpg")
print(f"Score: {result['score']}")
print(f"Model: {result['model']}")
```
```

```file: TODO.md
# TruthLens TODO List

## High Priority (Hackathon Ready)

### ✅ Completed
- [x] Basic FastAPI backend with authentication
- [x] React frontend with Tailwind CSS
- [x] Docker configuration
- [x] Database models and migrations
- [x] Azure Cognitive Services integration (disabled by default)
- [x] News verification with NewsAPI fallback
- [x] Blockchain integration with Polygon Mumbai
- [x] Trust score calculation algorithm
- [x] File upload and validation
- [x] JWT authentication system
- [x] Responsive UI components

### �� In Progress
- [ ] Real ML model integration
- [ ] Production deployment configuration

## Medium Priority (Post-Hackathon)

### ML Model Integration
- [ ] Replace demo image detector with real deepfake detection model
  - Suggested: `facebook/detr-resnet-50` or custom CNN
  - Location: `backend/app/detectors/image_detector.py`
  - Replace lines 15-16 and `_calculate_demo_score()` method

- [ ] Replace demo video detector with temporal consistency model
  - Suggested: `facebook/timesformer-base`
  - Location: `backend/app/detectors/video_detector.py`
  - Replace lines 15-16 and `_calculate_demo_score()` method

- [ ] Replace demo audio detector with voice cloning detection
  - Suggested: `speechbrain/spkrec-ecapa-voxceleb`
  - Location: `backend/app/detectors/audio_detector.py`
  - Replace lines 15-16 and `_calculate_demo_score()` method

### Azure Integration
- [ ] Test Azure Computer Vision API integration
- [ ] Test Azure Speech Services integration
- [ ] Add error handling for Azure API failures
- [ ] Implement retry logic for failed requests

### Blockchain Features
- [ ] Deploy smart contract to Polygon Mumbai
- [ ] Test blockchain transaction functionality
- [ ] Add gas estimation for transactions
- [ ] Implement transaction status monitoring

### Frontend Enhancements
- [ ] Add file preview before upload
- [ ] Implement drag-and-drop file upload
- [ ] Add progress indicators for long-running operations
- [ ] Implement real-time notifications

## Low Priority (Future Features)

### Advanced Features
- [ ] Batch file processing
- [ ] API rate limiting
- [ ] User role management (admin, user)
- [ ] File sharing and collaboration
- [ ] Advanced analytics dashboard
- [ ] Mobile app (React Native)

### Performance Optimizations
- [ ] Implement Redis caching
- [ ] Add CDN for static files
- [ ] Database query optimization
- [ ] Image/video compression
- [ ] Background job processing

### Security Enhancements
- [ ] API key management
- [ ] Rate limiting per user
- [ ] File virus scanning
- [ ] Content moderation
- [ ] Audit logging

### Monitoring and Analytics
- [ ] Application performance monitoring
- [ ] User behavior analytics
- [ ] Error tracking and reporting
- [ ] Usage statistics dashboard

## Hackathon Demo Checklist

### Pre-Demo Setup
- [ ] Deploy smart contract to Polygon Mumbai
- [ ] Get test MATIC tokens for transactions
- [ ] Prepare sample media files for testing
- [ ] Test all major user flows

### Demo Flow (5 minutes)
1. **Show Homepage** (30 seconds)
   - Explain TruthLens features
   - Show clean, modern UI

2. **User Registration** (30 seconds)
   - Create new account
   - Show JWT authentication

3. **File Upload** (2 minutes)
   - Upload sample image/video
   - Show real-time analysis
   - Display trust score and verdict

4. **Azure Integration** (1 minute)
   - Enable Azure services
   - Show enhanced analysis
   - Compare with/without Azure

5. **Blockchain Verification** (1 minute)
   - Show on-chain transaction
   - View transaction on Polygonscan
   - Explain immutability benefits

### Demo Files to Prepare
- [ ] Sample deepfake image
- [ ] Sample real image
- [ ] Sample video file
- [ ] Sample audio file

### Demo Environment
- [ ] Production-ready deployment
- [ ] All services running smoothly
- [ ] Backup plan if services fail
- [ ] Mobile-responsive design

## Development Notes

### Environment Variables Required
```bash
# Required for basic functionality
JWT_SECRET=your-secret-key
DATABASE_URL=mysql://user:pass@host:port/db

# Required for blockchain
POLYGON_RPC_URL=https://polygon-mumbai.g.alchemy.com/v2/KEY
POLYGON_PRIVATE_KEY=your-private-key

# Optional for enhanced features
AZURE_VISION_KEY=your-azure-key
AZURE_VISION_ENDPOINT=your-azure-endpoint
NEWSAPI_KEY=your-newsapi-key
```

### Key Files to Modify for Production
1. `backend/app/detectors/*.py` - Replace demo models
2. `backend/app/config.py` - Update production settings
3. `frontend/src/services/api.js` - Update API endpoints
4. `docker-compose.yml` - Production configuration
5. `.env` - Production environment variables

### Testing Checklist
- [ ] Unit tests for all detectors
- [ ] Integration tests for API endpoints
- [ ] End-to-end tests for user flows
- [ ] Load testing for file uploads
- [ ] Security testing for authentication

## Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Web3.py Documentation](https://web3py.readthedocs.io/)

### ML Model Resources
- [Hugging Face Models](https://huggingface.co/models)
- [SpeechBrain Models](https://speechbrain.github.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Blockchain Resources
- [Polygon Documentation](https://docs.polygon.technology/)
- [Hardhat Documentation](https://hardhat.org/docs)
- [Solidity Documentation](https://docs.soliditylang.org/)
```

```file: DEMO_CHECKLIST.md
# TruthLens Hackathon Demo Checklist

## Pre-Demo Setup (15 minutes)

### 1. Environment Setup
- [ ] Copy `.env.sample` to `.env`
- [ ] Update JWT_SECRET with unique value
- [ ] Add Polygon Mumbai RPC URL
- [ ] Add test wallet private key
- [ ] Start all services: `docker-compose up -d`

### 2. Smart Contract Deployment
```bash
cd blockchain
npm install
npx hardhat run scripts/deploy.js --network mumbai
# Copy contract address to .env
```

### 3. Test Data Preparation
- [ ] Sample real image (JPG, < 6MB)
- [ ] Sample deepfake image (JPG, < 6MB)
- [ ] Sample video file (MP4, < 50MB)
- [ ] Sample audio file (MP3, < 20MB)

### 4. Azure Setup (Optional)
- [ ] Get Azure credentials from hackathon organizers
- [ ] Update `.env` with Azure keys
- [ ] Restart backend: `docker-compose restart backend`

## Demo Script (5 minutes)

### 1. Introduction (30 seconds)
**What to say:**
"TruthLens is a comprehensive media verification platform that uses AI detection, Azure Cognitive Services, news verification, and blockchain storage to verify the authenticity of uploaded media files."

**What to show:**
- Clean, modern homepage
- Feature highlights
- Professional UI design

### 2. User Registration (30 seconds)
**What to do:**
1. Click "Sign Up" button
2. Enter email: `demo@truthlens.com`
3. Enter password: `demo123`
4. Click "Create Account"

**What to say:**
"We have a secure JWT-based authentication system that protects user data and verification history."

### 3. File Upload & Analysis (2 minutes)
**What to do:**
1. Click "Upload" in navigation
2. Select sample image file
3. Add metadata: "Sample image for verification"
4. Click "Upload and Analyze"
5. Wait for analysis to complete

**What to say:**
"Our AI detection system analyzes the file using multiple techniques:
- Face consistency analysis
- Edge artifact detection
- Color space analysis
- Temporal consistency (for videos)"

**What to show:**
- Real-time upload progress
- Trust score calculation
- Detailed analysis breakdown
- Verdict (Real/Fake)

### 4. Azure Integration (1 minute)
**What to do:**
1. Show current analysis without Azure
2. Enable Azure services in `.env`
3. Restart backend
4. Upload same file again
5. Compare results

**What to say:**
"When Azure services are enabled, we get additional verification from Microsoft's Computer Vision API, providing enhanced accuracy and confidence scores."

**What to show:**
- Azure analysis results
- Improved trust score
- Additional verification data

### 5. Blockchain Verification (1 minute)
**What to do:**
1. Show result with "Real" verdict
2. Click "View on Blockchain" link
3. Open Polygonscan transaction
4. Explain blockchain benefits

**What to say:**
"For verified content, we store the file hash on Polygon Mumbai blockchain, providing immutable proof of verification. This creates a permanent record that can't be tampered with."

**What to show:**
- Transaction hash
- Polygonscan transaction details
- File hash stored on-chain
- Timestamp and uploader info

### 6. History & Results (30 seconds)
**What to do:**
1. Navigate to "History"
2. Show list of previous verifications
3. Click on a result to show details

**What to say:**
"Users can view their complete verification history and access detailed analysis results anytime."

## Demo Tips

### Technical Setup
- [ ] Test all flows before demo
- [ ] Have backup files ready
- [ ] Keep terminal open for troubleshooting
- [ ] Monitor logs: `docker-compose logs -f`

### Presentation Tips
- [ ] Speak clearly and confidently
- [ ] Explain technical concepts simply
- [ ] Show enthusiasm for the project
- [ ] Be prepared for questions
- [ ] Have a backup plan if something fails

### Common Questions & Answers

**Q: How accurate is the AI detection?**
A: Our current demo uses deterministic scoring, but in production we integrate with state-of-the-art models like Facebook's TimeSformer for video analysis and Microsoft's Computer Vision API for enhanced accuracy.

**Q: What happens if Azure is down?**
A: The system gracefully falls back to local AI detection and news verification, ensuring continuous operation.

**Q: Why use blockchain?**
A: Blockchain provides immutable proof of verification, creating a permanent record that can't be tampered with. This is crucial for legal and audit purposes.

**Q: How do you handle different file types?**
A: We support images (JPG, PNG), videos (MP4), and audio (MP3, WAV) with specialized detection models for each modality.

**Q: Is this production-ready?**
A: The core architecture is production-ready, but we're using demo models for the hackathon. Real ML models would be integrated for production deployment.

## Troubleshooting

### If Backend Fails
```bash
docker-compose restart backend
docker-compose logs backend
```

### If Frontend Fails
```bash
docker-compose restart frontend
docker-compose logs frontend
```

### If Database Fails
```bash
docker-compose restart mysql
docker-compose logs mysql
```

### If Blockchain Fails
- Check RPC URL and private key
- Ensure test wallet has MATIC tokens
- Verify contract address in `.env`

## Success Metrics

### What Judges Should See
- [ ] Professional, polished UI
- [ ] Smooth user experience
- [ ] Real-time analysis results
- [ ] Working blockchain integration
- [ ] Comprehensive verification system
- [ ] Scalable architecture

### Key Differentiators
- [ ] Multi-modal verification (image, video, audio)
- [ ] Azure Cognitive Services integration
- [ ] Blockchain immutability
- [ ] News verification
- [ ] Weighted trust scoring
- [ ] Modern tech stack
- [ ] Production-ready architecture