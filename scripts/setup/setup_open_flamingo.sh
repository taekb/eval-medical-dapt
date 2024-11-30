#!/bin/bash

# Create conda environment and upgrade pip
conda create --name open-flamingo-env python=3.10 -y # Modify environment name as desired
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate open-flamingo-env
conda install -y pip
pip install --upgrade pip

conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install open-clip-torch==2.24.0
pip install transformers==4.38.2
pip install bitsandbytes
pip install open-flamingo[all]
pip install accelerate==0.28.0
pip install datasets
pip install wandb
pip install einops
pip install einops_exts
pip install h5py
pip install ipykernel ipywidgets
pip install hydra-core
pip install scikit-learn
pip install matplotlib
pip install seaborn

# Clone flash-attention repository and install package
cd ../../../
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation