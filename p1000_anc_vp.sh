#!/bin/bash

#SBATCH -J P1000_anc_vp
#SBATCH -A MLMI-USER-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:35:00
#SBATCH --mail-type=None
##SBATCH --no-requeue
#SBATCH -p ampere

options='--save_name=P1000_anc_vp --model=ddpm --predictor=AncestralSampling --corrector=None --num_gen_batches=10 --checkpoint=checkpoints/vp/cifar10_ddpm/checkpoint_26'

. /etc/profile.d/modules.sh
module purge
module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"
conda activate score_sde_env
module load cuda/11.8
module load cudnn/8.9_cuda-11.8

RUN_NAME=P1000_anc_vp

mkdir -p logs
mkdir -p generated

JOBID=$SLURM_JOB_ID
LOG=logs/"$RUN_NAME".log_$JOBID.txt
ERR=logs/"$RUN_NAME".err_$JOBID.txt

echo "Time: `date`" >> $LOG

python run.py $options >> $LOG 2>> $ERR 

echo "Time: `date`" >> $LOG