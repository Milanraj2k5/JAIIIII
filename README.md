# TruthLens - Media Verification Platform

TruthLens is a comprehensive media verification platform that uses AI detection, Azure Cognitive Services, news verification, and blockchain storage to verify the authenticity of uploaded media files.

## Features

- **AI Detection**: Local ML models for image, video, and audio deepfake detection
- **Azure Integration**: Optional Azure Cognitive Services for enhanced analysis
- **News Verification**: Cross-reference content with news sources
- **Blockchain Storage**: Store verified file hashes on Polygon Mumbai
- **Trust Scoring**: Weighted algorithm combining all verification methods
- **User Authentication**: JWT-based authentication system
- **Modern UI**: React frontend with Tailwind CSS

## Quick Start

### Prerequisites

- Node.js >= 18
- Python 3.10+
- Docker and Docker Compose
- Git

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd TruthLens
cp .env.sample .env
```

### 2. Configure Environment

Edit `.env` file with your settings:

```bash
# Required: Change these values
JWT_SECRET=your-unique-secret-key-here
POLYGON_RPC_URL=https://polygon-mumbai.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
POLYGON_PRIVATE_KEY=your-test-wallet-private-key

# Optional: Add during hackathon
AZURE_VISION_KEY=your-azure-vision-key
AZURE_VISION_ENDPOINT=your-azure-vision-endpoint
NEWSAPI_KEY=your-newsapi-key
```

### 3. Start with Docker

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 5. Test the Application

```bash
# Run smoke test
python scripts/smoke_test.py

# Or test manually
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "email=test@example.com&password=test123"
```

## Hackathon Setup

### Enable Azure Services

1. Get Azure credentials from hackathon organizers
2. Update `.env` file:
   ```bash
   AZURE_VISION_KEY=your-azure-key
   AZURE_VISION_ENDPOINT=your-azure-endpoint
   AZURE_SPEECH_KEY=your-speech-key
   AZURE_SPEECH_ENDPOINT=your-speech-endpoint
   ```
3. Restart services: `docker-compose r