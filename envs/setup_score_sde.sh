#!/bin/bash
set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/11.2
module load anaconda/3.2019-10

eval "$(conda shell.bash hook)"

ENV_PATH="/rds/user/$crsid/hpc-work/MLMI4/envs/score_sde_env"
rm -rf "$ENV_PATH" 2>/dev/null
conda create -p "$ENV_PATH" python=3.8 -y
conda activate "$ENV_PATH"

python -m pip install --upgrade pip

echo "Installing packages..."
python -m pip install \
    ml-collections==0.1.0 \
    tensorflow-gan==2.0.0 \
    tensorflow_io==0.17.1 \
    tensorflow_datasets==3.1.0 \
    tensorflow==2.4.0 \
    tensorflow-addons==0.12.0 \
    tensorboard==2.4.0 \
    absl-py==0.10.0 \
    flax==0.3.1 \
    jax==0.2.8 \
    tensorflow-probability==0.12.2 \
    numpy==1.19.5 \
    six==1.15.0 \
    termcolor==1.1.0 \
    wrapt==1.12.1 \
    protobuf==3.17.3

python -m pip install -U 'jaxlib==0.1.60+cuda111' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install scipy pillow matplotlib pyyaml imageio

echo "Done. Activate with: source envs/README.SCORE_SDE.activate"
