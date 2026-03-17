#!/bin/bash

#SBATCH -J Gen
#SBATCH -A MLMI-USER-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=None
##SBATCH --no-requeue
#SBATCH -p ampere

options='--save_name=P3 --corrector=None --sampler_steps=3 --sampler=ode'

. /etc/profile.d/modules.sh
module purge
module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"
conda activate score_sde_env
module load cuda/11.8
module load cudnn/8.9_cuda-11.8

RUN_NAME=P3

mkdir -p logs
mkdir -p generated

JOBID=$SLURM_JOB_ID
LOG=logs/"$RUN_NAME".log_$JOBID.txt
ERR=logs/"$RUN_NAME".err_$JOBID.txt

echo "Time: `date`" >> $LOG

python run.py $options >> $LOG 2>> $ERR 

echo "Time: `date`" >> $LOG