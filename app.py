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
from flask import Flask, request, jsonify
from flask_openapi3 import OpenAPI, Info, Tag
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

def _strip(v: str) -> str:
    if v and v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    return v or ""

GITHUB_TOKEN           = _strip(os.getenv("GITHUB_TOKEN"))
OWNER_REPO             = _strip(os.getenv("OWNER_REPO", "remla25-team14/model-training"))
ARTIFACT_ID            = _strip(os.getenv("ARTIFACT_ID", "3143858901"))
VECT_FILE_NAME_IN_ZIP  = _strip(os.getenv("VECT_FILE_NAME_IN_ZIP", "c1_BoW_v1.pkl"))
MODEL_FILE_NAME_IN_ZIP = _strip(os.getenv("MODEL_FILE_NAME_IN_ZIP", "c2_Classifier_v1.pkl"))
LOCAL_MODEL_CACHE_PATH = _strip(os.getenv("LOCAL_MODEL_CACHE_PATH", "model_cache"))
PORT                   = int(os.getenv("PORT", 5000))

with open("VERSION") as f:
    SERVICE_VERSION = f.read().strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Starting model-service v{SERVICE_VERSION}")

info = Info(title="Sentiment Analysis API", version=SERVICE_VERSION)
app = OpenAPI(__name__,
              info=info,
              doc_prefix="/"  
             )

sentiment_tag = Tag(name="sentiment", description="Sentiment analysis operations")
version_tag = Tag(name="version", description="API version information")
docs_tag = Tag(name="docs", description="API documentation") 

class VersionResponse(BaseModel):
    model_version: str = Field(..., description="Current model service version")

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

classifier = None
vectorizer = None

def download_and_extract_artifact(owner_repo, artifact_id, token, extract_dir, retries=3, delay=5):
    if not token:
        logging.error("GITHUB_TOKEN not set")
        return False
    api_url = f"https://api.github.com/repos/{owner_repo}/actions/artifacts/{artifact_id}/zip"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    os.makedirs(extract_dir, exist_ok=True)
    tmp = os.path.join(extract_dir, "artifact.zip")
    
    # Check cache validity - if files exist and were modified less than 24 hours ago
    vec_path = os.path.join(extract_dir, VECT_FILE_NAME_IN_ZIP)
    mdl_path = os.path.join(extract_dir, MODEL_FILE_NAME_IN_ZIP)
    
    if os.path.exists(vec_path) and os.path.exists(mdl_path):
        vec_age = time.time() - os.path.getmtime(vec_path)
        mdl_age = time.time() - os.path.getmtime(mdl_path)
        one_day = 86400  # 24 hours in seconds
        
        if vec_age < one_day and mdl_age < one_day:
            logging.info("Using cached model files (less than 24 hours old)")
            return True
    
    for attempt in range(1, retries+1):
        try:
            r = requests.get(api_url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            with zipfile.ZipFile(tmp, "r") as z:
                z.extractall(extract_dir)
            os.remove(tmp)
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(delay)
    logging.error("Could not download artifact")
    return False

def initialize_model():
    global classifier, vectorizer
    vec_path = os.path.join(LOCAL_MODEL_CACHE_PATH, VECT_FILE_NAME_IN_ZIP)
    mdl_path = os.path.join(LOCAL_MODEL_CACHE_PATH, MODEL_FILE_NAME_IN_ZIP)

    if not (os.path.exists(vec_path) and os.path.exists(mdl_path)):
        if not download_and_extract_artifact(OWNER_REPO, ARTIFACT_ID, GITHUB_TOKEN, LOCAL_MODEL_CACHE_PATH):
            logging.error("Artifact fetch/extract failed")
            return

    try:
        vectorizer = joblib.load(vec_path)
        logging.info(f"Loaded vectorizer from {vec_path}")
    except Exception as e:
        logging.error(f"Vectorizer load error: {e}")

    try:
        classifier = joblib.load(mdl_path)
        logging.info(f"Loaded classifier from {mdl_path}")
    except Exception as e:
        logging.error(f"Classifier load error: {e}")

try:
    from libml.data_preprocessing import preprocess_reviews
except ImportError:
    logging.error("libml import failed; using passthrough")
    def preprocess_reviews(df):
        return df["Review"].tolist()

initialize_model()

@app.get("/version", tags=[version_tag], responses={"200": VersionResponse})
def version():
    """
    Get the current model service version.
    """
    return jsonify({"model_version": SERVICE_VERSION})

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
    feedback_file = os.path.join(LOCAL_MODEL_CACHE_PATH, "feedback.json")
    
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

@app.get("/openapi.json", tags=[docs_tag])
def get_openapi_spec():
    """
    Get the OpenAPI specification for this API.
    
    This endpoint returns the OpenAPI JSON specification that describes all
    available endpoints, request parameters, and response formats.
    """
    return jsonify(app.get_openapi())

@app.get("/", tags=[docs_tag])
def redirect_to_docs():
    """
    Redirect to API documentation
    """
    from flask import redirect
    return redirect("/swagger")

if __name__ == "__main__":
    logging.info(f"Serving on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)