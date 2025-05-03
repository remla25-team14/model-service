import os
import time
import logging
import zipfile
import joblib
import requests

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd

# --- Load .env and strip any surrounding quotes ---
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

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

with open("VERSION") as f:
    SERVICE_VERSION = f.read().strip()
logging.info(f"Starting model-service v{SERVICE_VERSION}")


# --- Flask app & globals ---
app = Flask(__name__)
classifier = None
vectorizer = None

# --- Download & extract artifact ZIP safely ---
def download_and_extract_artifact(
    owner_repo: str,
    artifact_id: str,
    token: str,
    extract_dir: str,
    retries: int = 3,
    delay: int = 5
) -> bool:
    """Downloads a GitHub Actions artifact ZIP and extracts its files."""
    if not token:
        logging.error("GITHUB_TOKEN not set. Cannot authenticate to GitHub API.")
        return False

    api_url = f"https://api.github.com/repos/{owner_repo}/actions/artifacts/{artifact_id}/zip"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    os.makedirs(extract_dir, exist_ok=True)
    temp_zip = os.path.join(extract_dir, "artifact.zip")

    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Downloading artifact {artifact_id} (attempt {attempt})…")
            resp = requests.get(api_url, headers=headers, stream=True, timeout=60)
            resp.raise_for_status()

            # Write ZIP to disk
            with open(temp_zip, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Downloaded artifact to {temp_zip}")

            # Extract all files
            with zipfile.ZipFile(temp_zip, "r") as z:
                z.extractall(extract_dir)
                logging.info(f"Extracted ZIP into {extract_dir}")

            # Remove the ZIP after closing it
            os.remove(temp_zip)
            return True

        except (requests.RequestException, zipfile.BadZipFile) as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            # Cleanup partial file
            if os.path.exists(temp_zip):
                try: os.remove(temp_zip)
                except OSError: pass

            if attempt < retries:
                logging.info(f"Retrying in {delay}s…")
                time.sleep(delay)
            else:
                logging.error("Exceeded max retries for artifact download.")
                return False

    return False

# --- Initialize model & vectorizer ---
def initialize_model():
    global classifier, vectorizer

    vec_path = os.path.join(LOCAL_MODEL_CACHE_PATH, VECT_FILE_NAME_IN_ZIP)
    pkl_path = os.path.join(LOCAL_MODEL_CACHE_PATH, MODEL_FILE_NAME_IN_ZIP)

    # Download & extract if missing
    if not (os.path.exists(vec_path) and os.path.exists(pkl_path)):
        success = download_and_extract_artifact(
            OWNER_REPO,
            ARTIFACT_ID,
            GITHUB_TOKEN,
            LOCAL_MODEL_CACHE_PATH
        )
        if not success:
            logging.error("Artifact fetch/extract failed.")
            return

    # Load CountVectorizer
    try:
        vectorizer = joblib.load(vec_path)
        logging.info(f"CountVectorizer loaded from {vec_path}")
    except Exception as e:
        logging.error(f"Failed to load CountVectorizer: {e}")
        vectorizer = None

    # Load classifier
    try:
        classifier = joblib.load(pkl_path)
        logging.info(f"Classifier loaded from {pkl_path}")
    except Exception as e:
        logging.error(f"Failed to load classifier: {e}")
        classifier = None

# --- Preprocessing import ---
try:
    from libml.data_preprocessing import preprocess_reviews
except ImportError:
    logging.error("Could not import preprocess_reviews from libml; defining dummy passthrough.")
    def preprocess_reviews(df: pd.DataFrame):
        return df["Review"].tolist()

# --- Health check endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    ready = classifier is not None and vectorizer is not None
    status_code = 200 if ready else 503
    return jsonify({
        "status": "ok" if ready else "error",
        "model_loaded": ready,
        "message": None if ready else "Model or vectorizer not loaded"
    }), status_code

# --- Prediction endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    if classifier is None or vectorizer is None:
        return jsonify({"error": "Model not ready"}), 503
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        # Wrap input in DataFrame for your preprocess_reviews
        df = pd.DataFrame({"Review": [text]})
        processed_list = preprocess_reviews(df)
        processed = processed_list[0]

        # Vectorize → dense array for GaussianNB
        X_sparse = vectorizer.transform([processed])
        X = X_sparse.toarray()

        # Predict
        label = classifier.predict(X)[0]
        return jsonify({"sentiment": str(label)})

    except Exception as e:
        logging.error("Prediction error:", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- App startup ---
if __name__ == "__main__":
    logging.info("Initializing model and vectorizer…")
    initialize_model()
    logging.info(f"Starting Flask server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
