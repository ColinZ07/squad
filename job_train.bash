#!/bin/bash
#SBATCH --job-name=job_squad_train
#SBATCH --output=job_squad_train.txt
#SBATCH -p owners
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH -C GPU_MEM:16GB
#SBATCH -t 10:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/users/chenkaim/miniconda3/lib/
source activate squad

cd /scratch/users/chenkaim/personal/squad/

python train_QANet.py -n QANet

