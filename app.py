import os
import time
import logging
import zipfile
import uuid

import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ─── Load config & version ─────────────────────────────────────
load_dotenv()

def _strip(v: str) -> str:
    if v and v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    return v or ""

GITHUB_TOKEN           = _strip(os.getenv("GITHUB_TOKEN"))
OWNER_REPO             = _strip(os.getenv("OWNER_REPO", "remla25-team14/model-training"))
ARTIFACT_ID            = _strip(os.getenv("ARTIFACT_ID", "3053668556"))
VECT_FILE_NAME_IN_ZIP  = _strip(os.getenv("VECT_FILE_NAME_IN_ZIP", "c1_BoW_v1.pkl"))
MODEL_FILE_NAME_IN_ZIP = _strip(os.getenv("MODEL_FILE_NAME_IN_ZIP", "c2_Classifier_v1.pkl"))
LOCAL_MODEL_CACHE_PATH = _strip(os.getenv("LOCAL_MODEL_CACHE_PATH", "model_cache"))
PORT                   = int(os.getenv("PORT", 5000))

with open("VERSION") as f:
    SERVICE_VERSION = f.read().strip()

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Starting model-service v{SERVICE_VERSION}")

# ─── Flask App ──────────────────────────────────────────────────
app = Flask(__name__)
classifier = None
vectorizer = None

# ─── Download & extract (unchanged) ────────────────────────────
def download_and_extract_artifact(owner_repo, artifact_id, token, extract_dir, retries=3, delay=5):
    if not token:
        logging.error("GITHUB_TOKEN not set")
        return False
    api_url = f"https://api.github.com/repos/{owner_repo}/actions/artifacts/{artifact_id}/zip"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    os.makedirs(extract_dir, exist_ok=True)
    tmp = os.path.join(extract_dir, "artifact.zip")
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

# ─── Model Initialization ───────────────────────────────────────
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

# ─── Version Endpoint ──────────────────────────────────────────
@app.route("/version", methods=["GET"])
def version():
    return jsonify({"model_version": SERVICE_VERSION})

# ─── Analyze Endpoint ──────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    if classifier is None or vectorizer is None:
        return jsonify({"error": "Model not ready"}), 503

    data = request.get_json()
    review = data.get("review")
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

# ─── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info(f"Serving on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
