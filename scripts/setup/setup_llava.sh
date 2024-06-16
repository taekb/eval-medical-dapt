#!/bin/bash

# Reference: https://github.com/haotian-liu/LLaVA/tree/main#install

# Create conda environment and upgrade pip
conda create --name llava-env python=3.10 -y # Modify environment name as desired
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate llava-env
conda install -y pip
pip install --upgrade pip

conda install nvidia/label/cuda-12.1.0::cuda-toolkit

# Clone LLaVA repository and install package
cd ../../../
git clone git@github.com:haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Return to main repository
cd ../eval-vlm

# Install additional dependencies
pip install -r llava-requirements.txt
pip install --upgrade peft --no-dependencies

# Clone flash-attention repository and install package
# NOTE: If 'invalid cross-device link' error occurs, refer to this:
# https://github.com/Dao-AILab/flash-attention/issues/598#issuecomment-1784996156
cd ../../../
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation