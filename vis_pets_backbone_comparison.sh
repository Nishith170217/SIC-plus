#!/bin/bash
#SBATCH --job-name=pets_bb_vis
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

module load python/3.12-conda
conda activate sic-voc

cd /home/woody/rlvl/rlvl171v/SIC

DATA_DIR=/home/woody/rlvl/rlvl171v/data/oxford-iiit-pet
OUTPUT_DIR=/home/woody/rlvl/rlvl171v/SIC/visualizations/pets/backbone_comparison

python vis_pets_sic.py \
    --data_dir="$DATA_DIR" \
    --checkpoint=results/pets_full/best_model.pth \
    --output_dir="$OUTPUT_DIR" \
    --sample_indices 0 10 119 148 \
    --backbone=resnet50 \
    --n_shot=3 \
    --batch_size=32 \
    --support_batch_size=32 \
    --num_workers=8 \
    --percentile=95 \
    --smooth=15

python vis_pets_sic.py \
    --data_dir="$DATA_DIR" \
    --checkpoint=results/pets_densenet_full/best_model.pth \
    --output_dir="$OUTPUT_DIR" \
    --sample_indices 0 10 119 148 \
    --backbone=densenet121 \
    --n_shot=3 \
    --batch_size=32 \
    --support_batch_size=32 \
    --num_workers=8 \
    --percentile=95 \
    --smooth=15
