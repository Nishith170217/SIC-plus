#!/bin/bash
#SBATCH --job-name=sic_vis
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%j.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%j.err

module load python/3.12-conda
conda activate sic

cd /home/woody/rlvl/rlvl171v/SIC

python vis_sic.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/dogs/processed \
    --checkpoint=/home/woody/rlvl/rlvl171v/SIC/results/dogs_bs1/sic_dogs.pth \
    --batch_size=8 \
    --outdir=/home/woody/rlvl/rlvl171v/SIC/visualizations
