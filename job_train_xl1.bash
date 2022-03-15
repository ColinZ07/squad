#!/bin/bash
#SBATCH --job-name=job_qa_xl_1
#SBATCH --output=job_qa_xl_1.txt
#SBATCH -p owners
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH -C GPU_MEM:32GB
#SBATCH -t 24:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/users/chenkaim/miniconda3/lib/
source activate squad

cd /scratch/users/chenkaim/personal/squad/

python train_XL.py -n qaxl_1 --mem_len 64 --batch_size 64 --d_model 32 --d_inner 32 --batch_chunk 4 --lr 0.5
