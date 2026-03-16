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
python -m pip install "jax[cuda12]" --upgrade

echo "Installing modern ML packages..."
python -m pip install \
    "optax>=0.1.7" \
    "flax==0.10.7" \
    "orbax-checkpoint>=0.4.4,<0.11.0" \
    "ml-collections>=0.1.1" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "matplotlib>=3.7" \
    "ipykernel" \
    "importlib-resources"

echo "Installing TensorFlow stack (for score_sde)..."
python -m pip install \
    "tensorflow==2.13.0" \
    "tensorflow-datasets>=4.9.3" \
    "tensorflow-gan==2.1.0" \
    "tensorflow-addons==0.21.0" \
    "tensorflow-io==0.33.0" \
    "tensorboard==2.13.0" \
    "absl-py>=1.4.0"

echo "Installing utilities..."
python -m pip install pillow pyyaml imageio termcolor six

echo ""
echo "Done. Activate with: source envs/README.MLMI4.activate"
echo ""
