# Setup 

Project was developed using WSL2 (Ubuntu 18.04) on Windows 10. Assumes knowledge of the command line and Linux.
- Python version: 3.9

- [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/) to pull the repo and the large Tensorflow saved model.
    - This is to avoid having to train the model yourself.
- A Python Package manager:
    - [Anaconda Package Manager](https://www.anaconda.com/), feature-rich and supports different Python and CUDA versions (for ML with GPU). Needed for Windows
    - Or, [Pyenv](https://github.com/pyenv/pyenv), handles Python version management.
- [Pipenv](https://pipenv.pypa.io/en/latest/): specific Python environment manager we will use.
- [BentoML](https://github.com/bentoml/BentoML): Python Library and API to containerize, deploy and optimize inference for ML models.
- [Docker](https://www.docker.com/products/docker-desktop/), to use the deploy scripts
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html), to access AWS services from your command line

## Environment

(Optional) GPU support for Nvidia CUDA products. Anaconda prompt on Windows automatically handles this at the environment level.
- Project uses CUDA toolkit 11.2.0 and cuDNN 8.7.0

### Windows
1. Install [Nvidia Drivers](https://www.nvidia.com/download/index.aspx) for Windows.
2. Install Anaconda.
3. Create new Anaconda environment in this directory with:
    - `conda create -n <environment-name> python==3.9 -r requirements.txt`
4. Activate environment with the environment name you chose: `conda activate <environment name>`
5. Install pipenv with `pip install pipenv`
6. Run desired commands.

### WSL
1. Install [Nvidia Drivers](https://www.nvidia.com/download/index.aspx) for Windows.
2. Install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) version 11.x.x and [cuDNN](https://developer.nvidia.com/cudnn) version 8.x.x with:
    - [Official Tensorflow instructions](https://www.tensorflow.org/install/pip#windows-wsl2)
    - Or, download directly from Nvidia. Do not install any other Nvidia drivers
4. Install Anaconda, or Pipenv/Pyenv
5. Create environment in this directory with:
    - `conda create -n <environment-name> python==3.9 -r requirements.txt`
        - Run `conda activate <environment-name>` to use environment
        - Install Pipenv with Install pipenv with `pip install pipenv`
    - Or, create Pipenv with `make setup` (Runs `pipenv install`)
6. Run desired commands

### Linux
1. Install [Nvidia Drivers](https://www.nvidia.com/download/index.aspx) for Linux.
2. Install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) version 11.x.x
3. Install [cuDNN](https://developer.nvidia.com/cudnn) version 8.x.x
4. Install Anaconda, or Pipenv/Pyenv
5. Create new Anaconda environment in this directory with:
    - `conda create -n <environment-name> python==3.9 -r requirements.txt`
        - Run `conda activate <environment-name>` to use environment
        - Install Pipenv with Install pipenv with `pip install pipenv`
    - Or, create Pipenv with `make setup` (Runs `pipenv install`)
6. Run desired commands

## AWS CLI

To deploy the containerized version of this model to AWS ECS, we require a configured AWS CLI account.

After creating an [AWS account](https://aws.amazon.com/), configure your [CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
- This involves finding your access key and secret key, and running `aws configure` in the command line.