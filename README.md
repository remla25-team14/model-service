Using docker

docker build -t model-service .

docker run --rm --env-file .env -p 5000:5000 model-service

