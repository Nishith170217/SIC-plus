#!/bin/bash
#SBATCH --job-name=pets_dn_eval
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

module load python/3.12-conda
conda activate sic-voc

cd /home/woody/rlvl/rlvl171v/SIC

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_pets.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/oxford-iiit-pet \
    --checkpoint=/home/woody/rlvl/rlvl171v/SIC/results/pets_densenet_full/best_model.pth \
    --output=/home/woody/rlvl/rlvl171v/SIC/results/pets_densenet_full/evaluation.json \
    --batch_size=32 \
    --support_batch_size=32 \
    --num_workers=8 \
    --backbone=densenet121 \
    --n_shot=3
