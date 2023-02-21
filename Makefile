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
	@echo Creating ECR repository
	aws ecr get-login-password --region ${AWS_REGION} \
		| docker login \
			--username AWS \
			--password-stdin \
			${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
	aws ecr create-repository --repository-name ${AWS_ECR_REPO}

publish:
	$(eval AWS_USER_ID=$(shell aws sts get-caller-identity | jq -r '.UserId'))
	@echo
	@echo Pushing image ${MODEL_NAME}:${TAG} to ${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${AWS_ECR_REPO}
	docker tag ${MODEL_NAME}:${TAG} ${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${AWS_ECR_REPO}:${TAG}
	docker push ${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${AWS_ECR_REPO}:${TAG}

aws: repo publish
	$(eval AWS_USER_ID=$(shell aws sts get-caller-identity | jq -r '.UserId'))

	@ echo 
	@ echo Deploying stack for VPC/Subnet/Networking...
	aws cloudformation create-stack \
		--template-body file://$(shell pwd)/source/aws/vpc.yml \
		--parameters ParameterKey=ProjectName,ParameterValue="${MODEL_NAME}" \
		--stack-name ${MODEL_NAME}-vpc
	@echo Waiting for VPC stack to complete...
	aws cloudformation wait stack-create-complete --stack-name ${MODEL_NAME}-vpc

	@echo
	@echo Deploying stack for ECS Cluster...
	aws cloudformation create-stack \
		--template-body file://$(shell pwd)/source/aws/ecs-cluster.yml \
		--parameters ParameterKey=ProjectName,ParameterValue="${MODEL_NAME}" \
		--capabilities CAPABILITY_NAMED_IAM \
		--stack-name ${AWS_CLUSTER_NAME}
	@echo Waiting for ECS Cluster stack to complete...
	aws cloudformation wait stack-create-complete --stack-name ${AWS_CLUSTER_NAME}

	@echo
	@echo Deploying stack for task definition to ECS cluster
	aws cloudformation create-stack \
		--template-body file://$(shell pwd)/source/aws/app.yml \
		--parameters ParameterKey=ProjectName,ParameterValue="${MODEL_NAME}" \
			ParameterKey=UserId,ParameterValue="${AWS_USER_ID}" \
		--stack-name ${MODEL_NAME}-ecs-task-definition
	@echo Waiting for ECS Task Definition stack to complete...
	aws cloudformation wait stack-create-complete --stack-name ${MODEL_NAME}-ecs-task-definition

	@echo Deploy complete.

down:
	$(eval AWS_USER_ID=$(shell aws sts get-caller-identity | jq -r '.UserId'))

	@echo
	@echo "Deleting ECS Task Definition."
	aws cloudformation delete-stack --stack-name ${MODEL_NAME}-ecs-task-definition
	aws cloudformation wait stack-delete-complete --stack-name ${MODEL_NAME}-ecs-task-definition
	@echo "Waiting for delete confirmation..."
	@echo "Deleted."

	@echo "Deleting ECS Cluster."
	aws cloudformation delete-stack --stack-name ${AWS_CLUSTER_NAME}
	@echo "Waiting for delete confirmation..."
	@echo "Deleted."

	aws cloudformation wait stack-delete-complete --stack-name ${AWS_CLUSTER_NAME}
	@echo "Deleting VPC created for ECS cluster..."
	aws cloudformation delete-stack --stack-name ${MODEL_NAME}-vpc
	@echo "Waiting for delete confirmation..."
	aws cloudformation wait stack-delete-complete --stack-name ${MODEL_NAME}-vpc

	aws ecr get-login-password --region ${AWS_REGION} \
		| docker login \
			--username AWS \
			--password-stdin \
			${AWS_USER_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
	aws ecr delete-repository --force --repository-name ${AWS_ECR_REPO}


	@echo "All deploys deleted"
	
## Setup and cleanup
clean:
	rm ./*.h5
	rm ./*.db
	rm -rf ./kt_*/
	rm -rf ./checkpoints/
	bentoml delete ${MODEL_NAME}

setup:
	pipenv install --dev
