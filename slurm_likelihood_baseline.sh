#!/bin/bash

#SBATCH -J NLL
#SBATCH -A MLMI-gtk23-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=None
##SBATCH --no-requeue
#SBATCH -p ampere

options='--experiment_type=likelihood'

. /etc/profile.d/modules.sh
module purge
module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"
conda activate score_sde_env
module load cuda/11.8
module load cudnn/8.9_cuda-11.8

RUN_NAME=likelihood

mkdir -p logs
mkdir -p generated

JOBID=$SLURM_JOB_ID
LOG=logs/"$RUN_NAME".log_$JOBID.txt
ERR=logs/"$RUN_NAME".err_$JOBID.txt

echo "Time: `date`" >> $LOG

python run.py $options >> $LOG 2>> $ERR 

echo "Time: `date`" >> $LOG