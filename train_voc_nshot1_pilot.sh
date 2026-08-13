#!/bin/bash
#SBATCH --job-name=voc_ns1_pilot
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

module load python/3.12-conda
conda activate sic-voc

cd /home/woody/rlvl/rlvl171v/SIC

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_sic_voc.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/pascal-voc/VOCdevkit/VOC2007 \
    --results_dir=/home/woody/rlvl/rlvl171v/SIC/results/voc_nshot1_pilot \
    --epochs=1 \
    --batch_size=8 \
    --n_shot=1 \
    --accumulation_steps=2 \
    --num_workers=8 \
    --support_batch_size=32 \
    --val_batch_size=32 \
    --lr=0.0001 \
    --amp
