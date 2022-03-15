#!/bin/bash
#SBATCH --job-name=job_qa_xl_train_small
#SBATCH --output=job_qa_xl_train_small.txt
#SBATCH -p owners
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH -C GPU_MEM:16GB
#SBATCH -t 24:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/users/chenkaim/miniconda3/lib/
source activate squad

cd /scratch/users/chenkaim/personal/squad/

python train_XL.py -n QANet_XL_small --d_model 32 --d_inner 128 --num_head 4 --d_head 8 --mem_len 64 

