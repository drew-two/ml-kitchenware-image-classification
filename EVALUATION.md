# ML Zoomcamp Project Evaluation Criteria
# ML Zoomcamp 2022: Capstone 1

Self-tracking project progress.

1. Problem Description ([README.md](./README.md))
- [x] Described briefly
- [x] Described with enough context so problem and solution are clear

2. EDA ([eda.ipynb](./notebooks/eda.ipynb))
- [x] Basic EDA (looking at min-max values, checking for missing values)
- [x] Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis)
    - For images: analyzing the content of the images
    - For texts: frequent words, word clouds, etc

3. Model Training ([trainer.ipynb](./trainer.ipynb))
- [x] Trained only one model, no parameter tuning
- [x] Trained multiple models (linear and tree-based)
    - For neural networks: tried multiple variations - with dropout or without, with extra inner layers or without
    - **Note:** I explored CNNs as well as transformers, and their sizes. I believe this is sufficient.
- [x] Trained multiple models and tuned their parameters.
    - For neural networks: same as previous, but also with tuning: adjusting learning rate, dropout rate, size of the inner layer, etc.
    - **Note:** I explored image augmentation as well as learning rate. I believe this is sufficient.

4. Exporting notebook to script ([trainer.py](./trainer.py))
- [x] The logic for training the model is exported to a separate script

5. Reproducibility (Run `make train`)
- [x] It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data

6. Model deployment (Run `make run`)
- [x] Model is deployed (with Flask, BentoML or a similar framework)

7. Dependency and environment management 
- [x] Provided a file with dependencies (requirements.txt, pipfile, bentofile.yaml with dependencies, etc)
- [x] Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to activate the env

8. Containerization (Run `make build`)
- [x] Dockerfile is provided or a tool that creates a docker image is used (e.g. BentoML)
- [x] The application is containerized and the README describes how to build a contained and how to run it

9. Cloud deployment (Run `make publish`)
- [] Docs describe clearly (with code) how to deploy the service to cloud or kubernetes cluster (local or remote)
- [] There's code for deployment to cloud or kubernetes cluster (local or remote). There's a URL for testing - or video/screenshot of testing it