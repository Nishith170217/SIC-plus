#!/bin/bash
#SBATCH --job-name=voc_nshot_vis
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

set -euo pipefail

module load python/3.12-conda
conda activate sic-voc

cd /home/woody/rlvl/rlvl171v/SIC

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR=/home/woody/rlvl/rlvl171v/data/pascal-voc/VOCdevkit/VOC2007
OUTPUT_DIR=/home/woody/rlvl/rlvl171v/SIC/experiments/nshot/voc/explanations

python vis_voc_sic.py \
    --data_dir="$DATA_DIR" \
    --checkpoint=results/voc_nshot1/best_model.pth \
    --output_dir="$OUTPUT_DIR" \
    --sample_index=6 \
    --classes horse person \
    --max_classes=2 \
    --n_shot=1 \
    --threshold=0.5 \
    --batch_size=32 \
    --num_workers=8 \
    --percentile=95 \
    --smooth=15

python vis_voc_sic.py \
    --data_dir="$DATA_DIR" \
    --checkpoint=results/voc_full/best_model.pth \
    --output_dir="$OUTPUT_DIR" \
    --sample_index=6 \
    --classes horse person \
    --max_classes=2 \
    --n_shot=3 \
    --threshold=0.5 \
    --batch_size=32 \
    --num_workers=8 \
    --percentile=95 \
    --smooth=15

python vis_voc_sic.py \
    --data_dir="$DATA_DIR" \
    --checkpoint=results/voc_nshot5/best_model.pth \
    --output_dir="$OUTPUT_DIR" \
    --sample_index=6 \
    --classes horse person \
    --max_classes=2 \
    --n_shot=5 \
    --threshold=0.5 \
    --batch_size=32 \
    --num_workers=8 \
    --percentile=95 \
    --smooth=15
