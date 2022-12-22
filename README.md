# ML for Image Classification: Kitchenware Kaggle Competition

This repo explores simple CNNs from Keras applications, as well as larger and more powerful deep learning models such as image transformers to classify kitchenware and cutlery. Here we see exploratory data analysis (EDA), tuning different ML models, and hyperparameter search with Keras Tuner.
- Model is submitted to Kaggle as well as having a local and cloud deploy via [BentoML](https://www.bentoml.com/)

### Dataset

Dataset comes from [Kitchenware Classification Kaggle competition](https://www.kaggle.com/competitions/kitchenware-classification)

Includes of glasses, cups, plates, forks, knives and spoons.

### Technologies
- Python
- Anaconda
- Pipenv
- CUDA
- Pandas, NumPy
- Tensorflow/Keras
    - CNN models:
        - ConvNext
        - ConvNextV2
        - EfficientNet
    - Image transformer models:
        - MaxViT
        - DaViT
        - CAiT
        - DEiT\*
        - BEiT\**
- BentoML
- Docker
- AWS ECR
- AWS ECS

Starter notebook [keras-starter.ipynb](./source/notebooks/keras-starter.ipynb) from this [GitHub repo](https://github.com/DataTalksClub/kitchenware-competition-starter) for DataTalks.Club [ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code).

## Development Overview:
- Setup environment with Pipenv/Anaconda
- Setup GPU support with CUDA toolkit in WSL2 (Ubuntu)
- [EDA](./source/notebooks/eda.ipynb)
    - Visualize images
    - Visualize class imbalance
    - Visualize image sizes
    - Hypothesis for image augmentation
- Prepare Dataset (80/20 train/validation split)
- Explore different models for fine-tuning
    - Explore CNN models from Keras applications
    - Explore larger CNN models and Image Transformer models from [GitHub repo](https://github.com/leondgarse/)

## Setup

See [Setup instructions](./SETUP.md)

## Use

Refer to [Makefile](./Makefile). Everything you need to run will be in there
1. Run with `make run`
    - This performs all the training and building for you. You can see the Makefile to run these separately.
2. Testing:
    - Test from script:
        - Evaluate with the image in `testing/` with `make test`
    - Test by GUI:
        - Open http://localhost:3000/
        - Select the first option, to POST
        - Hit `Try it out` in the top right
        - In the drop down box, change `application/postscript` to `image/jpeg`
        - Upload the file `testing/0966.jpg` or any other file from the dataset.
        - Hit `Execute`, wait for your result

## Cloud Deploy

NOTE: As a Tensorflow image classification model, this is pretty large (~2.5GB image) and uses a decent amount of resources (~4GB memory).
- I can provide as URL for testing, just message me on the Slack (Andrew Katoch)
- Otherwise, the screenshots should be sufficient.

0. Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) and configure it.
1. Push image to cloud with `make publish`. This creates an AWS ECR repo and pushes to it
2. Install [ecs-cli](https://github.com/aws/amazon-ecs-cli#installing) with `make install-ecs-cli`.
3. Configure the ECS-cli profile with the same credentials as the AWS CLI: `./ecs-cli configure profile --access-key aws_access_key_id --secret-key aws_secret_access_key`
4. 

## Notebooks

0. EDA
    - [eda.ipynb](./notebooks/eda.ipynb)
1. Model experimentation
    - [keras-starter-deit-cait.ipynb](./notebooks/keras-starter-deit-cait.ipynb): explores the DEiT and CAiT transformer models
    - [keras-starter-efficientnet.ipynb](./notebooks/keras-starter-efficientnet.ipynb): explores EfficientNet CNN models
    - [keras-starter-maxvit.ipynb](./notebooks/keras-starter-maxvit.ipynb): explores the MaxViT transformer model
    - [keras-starter-swin.ipynb](./notebooks/keras-starter-swin.ipynb): explores the Swin Transformer model
    - [model-testing.ipynb](./notebooks/model-testing.ipynb): performs a search over some transformer and CNN models
2. Hyperparameter Search
    - [augmentation-test.ipynb](./notebooks/augmentation-test.ipynb): explores augmentation from ImageDataGenerator. None chosen
    - [randaug-testing.ipynb](./notebooks/randaug-testing.ipynb): explores augmentation using the RandAugment auto augmentation policy. Not chosen
    - [learning-rate-test.ipynb)](./notebooks/learning-rate-test.ipynb): searches for learning rate with large search space
    - [learning-rate-final.ipynb)](./notebooks/learning-rate-final.ipynb): searches for learning rate with narrower search space
3. Training
    - [kaggle-submission.ipynb](./notebooks/kaggle-submission.ipynb): submits trained BEiT model for Kaggle competition
    - [trainer.ipynb](./trainer.ipynb): training and bentoml capture of model for deployment (uses DEiT to save on space/time)
    - [trainer.py](./trainer.py): training script made from above notebook
4. Deploy
    - [service.py](./service.py): BentoML service used for Docker/ECS
    - [test_prediction.py](./testing/test_prediction.py): Script for testing service 

## Original README

A starter notebook for [the Kitchenware classification competition](https://www.kaggle.com/competitions/kitchenware-classification/) on Kaggle: [keras-starter.ipynb](keras-starter.ipynb)

In this notebook, we show how to:


- Download the data from Kaggle and unzip it
- Read the data
- Train an xception model (using the same code as in [ML Zoomcamp](http://mlzoomcamp.com))
- Make predictions
- Submit the results 

You can run this notebook in SaturnCloud:

<p align="center">
    <a href="https://app.community.saturnenterprise.io/dash/resources?recipeUrl=https://raw.githubusercontent.com/DataTalksClub/kitchenware-competition-starter/main/kitchenware-jupyter-recipe.json" target="_blank" rel="noopener">
        <img src="https://saturncloud.io/images/embed/run-in-saturn-cloud.svg" alt="Run in Saturn Cloud"/>
    </a>
</p>


Using the recipe:

- Download the credential file from Kaggle
- Put the content of the file to [SaturnCloud secrets](https://app.community.saturnenterprise.io/dash/o/community/secrets), save this secret as "kaggle" 
- Click on the button above to create a resource in SaturnCloud
- Verify that the kaggle secret is linked in the "secrets" tab
- Run the code and submit your predictions
- Improve the score

You can also see it as a video:


<a href="https://www.loom.com/share/c41e5691bd36414fa4df8de9c905cc58">
    <img src="https://user-images.githubusercontent.com/875246/206399525-097683c4-62bd-436b-815a-4ac8543502a9.png" />
</a>