MODEL_NAME=kitchenware-classification
TAG=latest

train:
	pipenv run python trainer.py

build: train
	pipenv run bentoml build
	pipenv run bentoml containerize ${MODEL_NAME}:${TAG} -t ${MODEL_NAME}:${TAG}

test:
	pipenv run python ./testing/test_prediction.py ./testing/0966.jpg "http://localhost:3000/predict_image"

serve: build
	pipenv run bentoml serve service.py:svc 

run: build
	docker run -it --rm -p 3000:3000 ${MODEL_NAME}:${TAG}

publish: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
