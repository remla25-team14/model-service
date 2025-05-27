# Restaurant Review Sentiment Analyzer API

This API automatically fetches and loads the latest sentiment analysis model from GitHub Actions artifacts on startup.

## Getting Started with Docker

### 1. Prepare Your Environment

- Create a `.env` file in the project root with your configuration variables such as:
    - `GITHUB_TOKEN`  
    Your GitHub token to access artifacts from the repository.  
    *Default:* (none)

    - `OWNER_REPO`  
    The GitHub repository containing the artifacts (format: owner/repo).  
    *Default:* `remla25-team14/model-training`

    - `ARTIFACT_ID`  
    The identifier of the artifact containing the model files.  
    *Default:* `3143858901`

    - `VECT_FILE_NAME_IN_ZIP`  
    The filename for the vectorizer inside the downloaded ZIP file.  
    *Default:* `c1_BoW_v1.pkl`

    - `MODEL_FILE_NAME_IN_ZIP`  
    The filename for the classifier inside the downloaded ZIP file.  
    *Default:* `c2_Classifier_v1.pkl`

    - `LOCAL_MODEL_CACHE_PATH`  
    The directory where the model files will be cached locally.  
    *Default:* `model_cache`

    - `PORT`  
    The port on which the service will listen.  
    *Default:* `5000`

### 2. Build the Docker Image

Run the following command in your terminal:

```
docker build -t model-service .
```

### 3. Run the Container

Start the container using:

```
docker run --rm --env-file .env -p 5000:5000 model-service
```

This will start the API at [http://localhost:5000](http://localhost:5000).

## API Documentation

Once the service is running, access the interactive API documentation using Swagger UI:

[http://localhost:5000/swagger](http://localhost:5000/swagger)