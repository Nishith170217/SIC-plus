#!/bin/bash
#SBATCH --job-name=dogs_support_select
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

python analysis/support_selection/evaluate_dogs_support_selection.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/dogs/processed \
    --checkpoint=/home/woody/rlvl/rlvl171v/SIC/results/dogs_bs1/sic_dogs.pth \
    --output_dir=/home/woody/rlvl/rlvl171v/SIC/experiments/support_selection/dogs \
    --batch_size=32 \
    --support_batch_size=32 \
    --num_workers=8 \
    --n_shot=3 \
    --random_seeds 0 1 2 3 4
