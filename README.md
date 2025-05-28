# model-service

Sentiment analysis API service with dynamic versioning.

## Quick Start

```bash
git clone https://github.com/remla25-team14/model-service.git
cd model-service
pip install -r requirements.txt
python test_model_service.py
```

## Usage

```bash
# Start the service
python app.py

# Check version
curl http://localhost:5000/version

# Analyze sentiment
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"review": "Great food!"}'
```

## Features

- ✅ Dynamic versioning from lib-version
- ✅ Text preprocessing with lib-ml
- ✅ Docker container with multi-arch support
- ✅ OpenAPI documentation

## Dependencies

- **lib-version@v0.1.0**: Dynamic versioning
- **lib-ml@0.1.6**: Text preprocessing

That's it! 🎉

