#!/bin/bash
#SBATCH --job-name=pets_errors_vis
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

python vis_pets_sic.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/oxford-iiit-pet \
    --checkpoint=/home/woody/rlvl/rlvl171v/SIC/results/pets_full/best_model.pth \
    --output_dir=/home/woody/rlvl/rlvl171v/SIC/visualizations/pets \
    --sample_indices 10 119 148 \
    --batch_size=32 \
    --support_batch_size=32 \
    --num_workers=8 \
    --n_shot=3 \
    --percentile=95 \
    --smooth=15
