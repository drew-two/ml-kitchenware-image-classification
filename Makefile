MODEL_NAME=kitchenware-classification

train:
	pipenv run python trainer.py

build: 
	pipenv run bentoml build
	sudo pipenv run bentoml containerize ${MODEL_NAME}:latest

test: local_deploy
	pipenv run python ./testing/tester.py ./testing/0966.jpg

local_deploy:


publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
