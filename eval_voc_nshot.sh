#!/bin/bash
#SBATCH --job-name=voc_nshot_eval
#SBATCH --array=0-2
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/woody/rlvl/rlvl171v/SIC/logs/%A_%a.out
#SBATCH --error=/home/woody/rlvl/rlvl171v/SIC/logs/%A_%a.err

module load python/3.12-conda
conda activate sic-voc

cd /home/woody/rlvl/rlvl171v/SIC

NSHOTS=(1 3 5)
RESULT_DIRS=(voc_nshot1 voc_full voc_nshot5)

NSHOT="${NSHOTS[$SLURM_ARRAY_TASK_ID]}"
RESULT_DIR="${RESULT_DIRS[$SLURM_ARRAY_TASK_ID]}"

python eval_voc.py \
    --data_dir=/home/woody/rlvl/rlvl171v/data/pascal-voc/VOCdevkit/VOC2007 \
    --checkpoint="/home/woody/rlvl/rlvl171v/SIC/results/${RESULT_DIR}/best_model.pth" \
    --output="/home/woody/rlvl/rlvl171v/SIC/results/${RESULT_DIR}/evaluation_nshot${NSHOT}.json" \
    --n_shot="${NSHOT}" \
    --batch_size=32 \
    --support_batch_size=32 \
    --num_workers=8 \
    --threshold=0.5
