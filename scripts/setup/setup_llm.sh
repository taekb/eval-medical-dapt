#!/bin/bash

# Create conda environment and upgrade pip
conda create --name llm-env python=3.10 -y # Modify environment name as desired
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate llm-env
conda install -y pip
pip install --upgrade pip

conda install -y nvidia::cuda-toolkit=12.1
conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.44.2
pip install bitsandbytes==0.43.3
pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install peft
pip install datasets
pip install wandb
pip install hydra-core
pip install scikit-learn
pip install ipykernel ipywidgets jupyterlab
pip install vllm
pip install langchain
pip install matplotlib
pip install seaborn

# Clone flash-attention repository and install package
# NOTE: If 'invalid cross-device link' error occurs, refer to this:
# https://github.com/Dao-AILab/flash-attention/issues/598#issuecomment-1784996156
cd ../../../
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation