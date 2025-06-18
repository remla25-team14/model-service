# model-service

Sentiment analysis API service with dynamic versioning and GitHub release-based model loading.

## Quick Start

```bash
git clone https://github.com/remla25-team14/model-service.git
cd model-service
pip install -r requirements.txt

# Set the model version to download from GitHub releases
export TRAINED_MODEL_VERSION=v0.1.3

# Test the service
python test_model_service.py

# Start the service
python app.py
```

## Usage

```bash
# Start the service
export TRAINED_MODEL_VERSION=v0.1.3
python app.py

# Check version (returns both service and model versions)
curl http://localhost:5000/version
# Response: {"service_version":"0.1.1","model_version":"v0.1.3"}

# Analyze sentiment
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"review": "Great food!"}'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAINED_MODEL_VERSION` | GitHub release tag to download model from | `v0.1.0` |
| `MODEL_CACHE_DIR` | Directory to cache downloaded models | `model_cache` |
| `VECT_FILE_NAME` | Vectorizer filename in release | `c1_BoW_Sentiment_Model.pkl` |
| `MODEL_FILE_NAME` | Classifier filename in release | `c2_Classifier_v1.pkl` |
| `PORT` | Service port | `5000` |

## Features

- ✅ Dynamic versioning from lib-version
- ✅ GitHub release-based model loading
- ✅ Text preprocessing with lib-ml
- ✅ Docker container with multi-arch support
- ✅ OpenAPI documentation
- ✅ Model caching by version
- ✅ Version endpoint shows both service and model versions

## Dependencies

- **lib-version@main**: Dynamic versioning
- **lib-ml@0.1.6**: Text preprocessing

## Model Loading

The service automatically downloads model artifacts from GitHub releases:
- Release URL: `https://github.com/remla25-team14/model-training/releases/tag/{TRAINED_MODEL_VERSION}`
- Required files: `c1_BoW_Sentiment_Model.pkl` (vectorizer) and `c2_Classifier_v1.pkl` (classifier)
- Models are cached in `{MODEL_CACHE_DIR}/{TRAINED_MODEL_VERSION}/`

### Model Version Requirements

- The `TRAINED_MODEL_VERSION` must match an existing release tag in the [model-training repository](https://github.com/remla25-team14/model-training/releases)
- The release must contain the exact filenames: `c1_BoW_Sentiment_Model.pkl` and `c2_Classifier_v1.pkl`
- If the release doesn't exist or is missing required files, the service will fail to start

### Version Endpoint

The `/version` endpoint returns both versions:
- **service_version**: The version of the model-service application (from lib-version)
- **model_version**: The version of the loaded ML model (from TRAINED_MODEL_VERSION)

Example response:
```json
{
  "service_version": "0.1.1",
  "model_version": "v0.1.3"
}
```

That's it! 🎉

