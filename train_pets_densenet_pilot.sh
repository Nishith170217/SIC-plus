#!/bin/bash
#SBATCH --job-name=pets_dn_pilot
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

module load python/3.12-conda
conda activate sic-voc

cd /home/woody/rlvl/rlvl171v/SIC

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_sic_pets.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/oxford-iiit-pet \
    --results_dir=/home/woody/rlvl/rlvl171v/SIC/results/pets_densenet_pilot \
    --epochs=1 \
    --batch_size=8 \
    --lr=0.001 \
    --n_way=30 \
    --n_shot=3 \
    --backbone=densenet121 \
    --accumulation_steps=4 \
    --num_workers=8 \
    --support_batch_size=32 \
    --val_batch_size=32 \
    --amp
