#!/bin/bash

ENV_NAME='score_sde_env'

module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"

conda create -n $ENV_NAME python=3.8.20 
conda activate $ENV_NAME

python -m pip install pip==20.2.4

pip install "jaxlib==0.1.59+cuda110" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements.txt