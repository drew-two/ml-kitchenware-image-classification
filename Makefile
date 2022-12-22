# Variables
MODEL_NAME=kitchenware-classification
TAG=latest
AWS_ECR_REPO=${MODEL_NAME}-repo
AWS_CLUSTER_NAME=${MODEL_NAME}-cluster

## Download Dataset
dataset: 
	kaggle competitions download -c kitchenware-classification
	mkdir data
	unzip kitchenware-classification.zip -d data > /dev/null
	rm kitchenware-classification.zip

## Training model and running locally with BentoML
train:
	pipenv run python ./source/trainer.py

bento:
	pipenv run python ./

build:
	cd ./source/; \
	pipenv run bentoml build; \
	pipenv run bentoml containerize ${MODEL_NAME}:${TAG} -t ${MODEL_NAME}:${TAG};

serve: bento build
	pipenv run bentoml serve service.py:svc 

run: bento build
	docker run -it --rm -p 3000:3000 ${MODEL_NAME}:${TAG}

test:
	pipenv run python ./source/test/test_prediction.py ./source/test/0966.jpg "http://localhost:3000/predict_image"

## Use AWS EC2 Container Registry, and Elastic Container Service to deploy
repo:
	$(eval AWS_REGION=$(shell aws configure get region))
	$(eval AWS_USER_ID=$(shell aws sts get-caller-identity | jq -r '.UserId'))
	aws ecr get-login-password --region ${AWS_REGION} \
		| docker login \
			--username AWS \
			--password-stdin \
			${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
	aws ecr create-repository --repository-name ${AWS_ECR_REPO}

publish:
	docker tag ${MODEL_NAME}:${TAG} ${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${AWS_ECR_REPO}:${TAG}
	docker push ${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${AWS_ECR_REPO}:${TAG}

aws: repo publish 
	# Create stack for VPC/Subnet/Networking
	aws cloudformation create-stack \
		--template-body file://$(pwd)/source/aws/vpc.yml \
		--parameters ParameterKey=ProjectName,ParameterValue="${MODEL_NAME}" \
		--stack-name ${MODEL_NAME}-vpc #\
		# | jq r '.StackID'
	# Stack for ECS Cluster
	aws cloudformation create-stack \
		--template-body file://$(pwd)/source/aws/ecs-cluster.yml \
		--parameters ParameterKey=ProjectName,ParameterValue="${MODEL_NAME}" \
		--capabilities CAPABILITY_NAMED_IAM \
		--stack-name ${AWS_CLUSTER_NAME}
	# Stack to deploy app to ECS cluster
	aws cloudformation create-stack \
		--template-body file://$(pwd)/source/aws/app.yml \
		--parameters ParameterKey=ProjectName,ParameterValue="${MODEL_NAME}" \
			ParameterKey=UserId,ParameterValue="${AWS_USER_ID}" \
		--stack-name ${MODEL_NAME}-task-definition

down:
	aws cloudformation delete-stack --stack-name ${MODEL_NAME}-task-definition
	aws cloudformation delete-stack --stack-name ${AWS_CLUSTER_NAME}
	aws cloudformation delete-stack --stack-name ${MODEL_NAME}-vpc

## Setup and cleanup
clean:
	rm ./*.h5
	rm ./*.db
	rm -rf ./kt_*/
	rm -rf ./checkpoints/
	bentoml delete ${MODEL_NAME}

setup:
	pipenv install --dev
