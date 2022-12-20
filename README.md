# Kitchenware Competition Image classification

Starter notebook [keras-starter.ipynb](./keras-starter.ipynb) from [this GitHub repo](https://github.com/DataTalksClub/kitchenware-competition-starter) by Alexey Grigorev.

- Using the starter notebook to learn how to load the competition data and make submissions, this repo explores other, larger and more powerful deep learning models to solve this problem.
- CNN models include:
    - ConvNext
    - ConvNextV2
    - EfficientNet
- Image transformer models include:
    - MaxViT
    - DaViT
    - BAiT
    - CAiT

Models not from Keras applications come from [this](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/beit) attention model GitHub repo by leondgarse.
- Information on original papers also found here.

## Dataset

Dataset comes from [Kitchenware Classification Kaggle competition](https://www.kaggle.com/competitions/kitchenware-classification).
- Can download dataset directly or after installing [Kaggle API](https://www.kaggle.com/docs/api)
    - You can run the commands at the beginning of the [starter notebook](./notebooks/keras-starter.ipynb), or: 
        ```
        kaggle competitions download -c kitchenware-classification
        mkdir data
        unzip kitchenware-classification.zip -d data > /dev/null
        rm kitchenware-classification.zip
        ```

## Setup

0. (Optional) GPU support:
    - Get appropriate CUDA v11.x.x from [Nvidia](https://developer.nvidia.com/cuda-11.2.2-download-archive).
        - (Linux) Add to your .bashrc: `echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/' >> ~/.bashrc`  
    - Install [cuDNN 8.1.0](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
    - NOTE: Need to install it this way rather than via Anaconda to work with Pipenv out of the box
1. Install pipenv and pyenv (recommended):
    - `pip install pipenv pyenv`
    - `make setup`
    - By default this installs the development packages. You can run `pipenv install` for just the deploy packages.

## Use

Can see EDA in [this notebook](./notebooks/eda.ipynb).



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
3. Final Drafts
- [kaggle-submission.ipynb](./notebooks/kaggle-submission.ipynb): submits trained BEiT model for Kaggle competition
- [trainer.ipynb](./trainer.ipynb): training and bentoml capture of model for deployment (uses DEiT to save on space/time)

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