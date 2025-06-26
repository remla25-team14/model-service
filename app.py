import os
import time
import logging
import zipfile
import uuid
import json
from typing import Dict, Optional, List, Any

import joblib
import pandas as pd
import requests
import pickle
from flask import Flask, request, jsonify
from flask_openapi3 import OpenAPI, Info, Tag
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from flask_openapi3.utils import get_operation
from apispec import APISpec

# ─── Load config & version ─────────────────────────────────────
load_dotenv()

def _strip(v: str) -> str:
    return v.strip() if v else ""

# Read token from Docker secret file if available
token_file = os.getenv("GITHUB_TOKEN_FILE")
if token_file and os.path.exists(token_file):
    with open(token_file, "r") as f:
        GITHUB_TOKEN = f.read().strip()
else:
    GITHUB_TOKEN = _strip(os.getenv("GITHUB_TOKEN"))

HOST = os.getenv("HOST", "0.0.0.0")
TRAINED_MODEL_VERSION = _strip(os.getenv("TRAINED_MODEL_VERSION", "v0.1.0"))
MODEL_SERVICE_IMAGE_TAG = _strip(os.getenv("MODEL_SERVICE_IMAGE_TAG"))
if not MODEL_SERVICE_IMAGE_TAG:
    try:
        with open("VERSION") as f:
            MODEL_SERVICE_IMAGE_TAG = f.read().strip()
    except Exception:
        MODEL_SERVICE_IMAGE_TAG = "unknown"
MODEL_CACHE_DIR = _strip(os.getenv("MODEL_CACHE_DIR", "model_cache"))
REPO = "remla25-team14/model-training"
VECT_FILE_NAME = _strip(os.getenv("VECT_FILE_NAME", "c1_BoW_Sentiment_Model.pkl"))
MODEL_FILE_NAME = _strip(os.getenv("MODEL_FILE_NAME", "c2_Classifier_v1.pkl"))
PORT = int(os.getenv("PORT", 5000))

# If TRAINED_MODEL_VERSION is 'latest', resolve to the latest release tag from GitHub
if TRAINED_MODEL_VERSION == "latest":
    logging.info("Resolving the 'latest' model release tag from GitHub...")
    try:
        api_url = f"https://api.github.com/repos/{REPO}/releases/latest"
        headers = {}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        resp = requests.get(api_url, headers=headers)
        resp.raise_for_status()  # Raise an exception for bad status codes
        
        TRAINED_MODEL_VERSION = resp.json()["tag_name"]
        logging.info(f"Successfully resolved 'latest' to release tag: {TRAINED_MODEL_VERSION}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch latest model release from GitHub: {e}")
        # Fallback to a known stable version if 'latest' fails
        TRAINED_MODEL_VERSION = "v0.1.5" 
        logging.warning(f"Falling back to known stable model version: {TRAINED_MODEL_VERSION}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while resolving the latest release: {e}")
        TRAINED_MODEL_VERSION = "v0.1.5"
        logging.warning(f"Falling back to known stable model version: {TRAINED_MODEL_VERSION}")

# Get version from lib-version
try:
    from libversion import VersionUtil
    SERVICE_VERSION = VersionUtil.get_version()
except ImportError:
    SERVICE_VERSION = "unknown"

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Starting model-service v{SERVICE_VERSION}")

# ─── Flask App & OpenAPI Setup ────────────────────────────────────
info = Info(title="Sentiment Analysis API", version=SERVICE_VERSION)
app = OpenAPI(__name__,
              info=info,
              doc_prefix="/"  
             )

sentiment_tag = Tag(name="sentiment", description="Sentiment analysis operations")
version_tag = Tag(name="version", description="API version information")
docs_tag = Tag(name="docs", description="API documentation") 

# ─── API Models (Pydantic) ────────────────────────────────────
class VersionResponse(BaseModel):
    service_version: str = Field(..., description="Current model service version")
    model_version: str = Field(..., description="Current loaded model version")

class ReviewRequest(BaseModel):
    review: str = Field(..., min_length=1, description="Restaurant review text to analyze")

class FeedbackRequest(BaseModel):
    review_id: str = Field(..., description="ID of the analyzed review")
    correct_sentiment: bool = Field(..., description="The correct sentiment (true=positive, false=negative)")

class AnalysisResponse(BaseModel):
    review_id: str = Field(..., description="Unique identifier for this analysis")
    review: str = Field(..., description="Original review text")
    sentiment: bool = Field(..., description="Predicted sentiment (true=positive, false=negative)")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0) of the prediction")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")

class FeedbackResponse(BaseModel):
    status: str = Field(..., description="Status of the feedback submission") 
    message: str = Field(..., description="Additional information about the feedback submission")

# ─── Global Model variables ─────────────────────────────────────
classifier = None
vectorizer = None

# ─── Model Download and Loading from GitHub Release ────────────
def download_from_github_release(version, asset_name, dest_path):
    url = f"https://github.com/{REPO}/releases/download/{version}/{asset_name}"
    logging.info(f"Downloading {asset_name} from {url}")
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    else:
        # This will now only happen if the secret is not mounted and the env var is not set
        logging.error("GITHUB_TOKEN not found.")

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(r.content)
        logging.info(f"Downloaded {asset_name} to {dest_path}")
    else:
        raise RuntimeError(f"Failed to download {asset_name} from {url} (status {r.status_code})")

def initialize_model():
    global classifier, vectorizer
    versioned_cache_dir = os.path.join(MODEL_CACHE_DIR, TRAINED_MODEL_VERSION)
    os.makedirs(versioned_cache_dir, exist_ok=True)
    model_path = os.path.join(versioned_cache_dir, MODEL_FILE_NAME)
    vec_path = os.path.join(versioned_cache_dir, VECT_FILE_NAME)

    # Download if missing
    if not os.path.exists(model_path):
        download_from_github_release(TRAINED_MODEL_VERSION, MODEL_FILE_NAME, model_path)
    if not os.path.exists(vec_path):
        download_from_github_release(TRAINED_MODEL_VERSION, VECT_FILE_NAME, vec_path)

    # Load model and vectorizer
    try:
        classifier = joblib.load(model_path)
        logging.info(f"Loaded classifier from {model_path}")
    except Exception as e:
        logging.error(f"Classifier load error: {e}")
        raise
    try:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        logging.info(f"Loaded vectorizer from {vec_path}")
    except Exception as e:
        logging.error(f"Vectorizer load error: {e}")
        raise

try:
    from libml.data_preprocessing import preprocess_reviews
except ImportError:
    logging.error("libml import failed; using passthrough")
    def preprocess_reviews(df):
        return df["Review"].tolist()

initialize_model()

# ─── Version Endpoint ──────────────────────────────────────────
@app.get("/version", tags=[version_tag], responses={"200": VersionResponse})
def version():
    """
    Get the current model service and model artifact version.
    """
    return jsonify({
        "service_version": MODEL_SERVICE_IMAGE_TAG,
        "model_version": TRAINED_MODEL_VERSION
    })

# ─── Analyze Endpoint ──────────────────────────────────────────
@app.post("/analyze", tags=[sentiment_tag], 
          responses={"200": AnalysisResponse, "400": ErrorResponse, "503": ErrorResponse})
def analyze(body: ReviewRequest):
    """
    Analyze the sentiment of a restaurant review.
    
    This endpoint processes restaurant review text and returns a sentiment prediction
    (positive or negative) along with a confidence score if available.
    """
    if classifier is None or vectorizer is None:
        return jsonify({"error": "Model not ready"}), 503

    review = body.review
    if not review:
        return jsonify({"error": "Missing 'review' field"}), 400

    # Preprocess and vectorize
    df = pd.DataFrame({"Review": [review]})
    processed = preprocess_reviews(df)[0]
    X = vectorizer.transform([processed]).toarray()

    # Predict
    sentiment = classifier.predict(X)[0]
    # Confidence (if classifier supports predict_proba)
    try:
        confs = classifier.predict_proba(X)[0]
        confidence = float(max(confs))
    except Exception:
        confidence = None

    return jsonify({
        "review_id": str(uuid.uuid4()),
        "review": review,
        "sentiment": bool(int(sentiment)),
        "confidence": confidence
    })

# ─── Feedback receiver on model-service ──────────────────────────
@app.post("/feedback", tags=[sentiment_tag],
          responses={"200": FeedbackResponse, "400": ErrorResponse})
def receive_feedback(body: FeedbackRequest):
    """
    Submit feedback on a sentiment analysis result.
    
    This endpoint allows users to provide feedback on whether the sentiment
    prediction was correct or not, which can be used to improve future models.
    """
    # Basic validation
    if not body.review_id:
        return jsonify({"error": "Missing 'review_id'"}), 400

    # Store the feedback for future model improvement
    feedback_file = os.path.join(MODEL_CACHE_DIR, "feedback.json")
    
    # Create or append to feedback file
    try:
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
            
        feedback_data.append({
            "review_id": body.review_id,
            "correct_sentiment": body.correct_sentiment,
            "timestamp": time.time()
        })
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f)
            
        logging.info(f"Feedback saved: {body.review_id}, correct={body.correct_sentiment}")
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
        
    # Ack back to sender
    return jsonify({"status": "success", "message": "Feedback saved for future model improvement"}), 200

# ─── OpenAPI Spec Endpoint ─────────────────────────────────────
@app.get("/openapi.json", tags=[docs_tag])
def get_openapi_spec():
    """
    Get the OpenAPI specification for this API.
    
    This endpoint returns the OpenAPI JSON specification that describes all
    available endpoints, request parameters, and response formats.
    """
    return jsonify(app.get_openapi())

# Add a redirect from the root to the Swagger UI
@app.get("/", tags=[docs_tag])
def redirect_to_docs():
    """
    Redirect to API documentation
    """
    from flask import redirect
    return redirect("/swagger")

# ─── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info(f"Serving on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT)
