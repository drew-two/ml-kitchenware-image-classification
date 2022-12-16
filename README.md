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

## Files

- [keras-starter-deit-cait.ipynb](./keras-starter-deit-cait.ipynb): explores the DEiT and CAiT transformer models
- [keras-starter-efficientnet.ipynb](./keras-starter-efficientnet.ipynb): explores EfficientNet CNN models
- [keras-starter-maxvit.ipynb](./keras-starter-maxvit.ipynb): explores the MaxViT transformer model
- [keras-starter-swin.ipynb](./keras-starter-swin.ipynb): explores the Swin Transformer model
- [model-testing.ipynb](./model-testing.ipynb): performs a search over some transformer and CNN models

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
