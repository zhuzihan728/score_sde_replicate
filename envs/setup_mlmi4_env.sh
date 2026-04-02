#!/bin/bash
set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1
module load cudnn/8.9_cuda-12.1
module load anaconda/3.2019-10

eval "$(conda shell.bash hook)"

ENV_PATH="/rds/user/$USER/hpc-work/MLMI4/envs/mlmi4_env"
rm -rf "$ENV_PATH" 2>/dev/null
conda create -p "$ENV_PATH" python=3.10 -y
conda activate "$ENV_PATH"

python -m pip install --upgrade pip

echo "Installing JAX with CUDA 12 support..."
python -m pip install -q "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing modern ML packages..."
python -m pip install -q \
    numpy \
    optax \
    flax \
    orbax-checkpoint \
    tensorflow \
    tensorflow_hub \
    ml_collections \
    scipy \
    matplotlib \
    importlib-resources \
    tensorflow_datasets \
    Pillow

echo ""
echo "Done. Activate with: source envs/README.MLMI4.activate"
echo ""
