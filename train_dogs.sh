#!/bin/bash
#SBATCH --job-name=sic_dogs
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

# Load environment
module load python/3.12-conda
conda activate sic

# Go to project directory
cd /home/woody/rlvl/rlvl171v/SIC

# Run training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sic.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/dogs/processed \
    --epochs=50 \
    --batch_size=8 \
    --n_way=30 \
    --accumulation_steps=4 \
    --resume \
    --results_dir=/home/woody/rlvl/rlvl171v/SIC/results/dogs_bs1

