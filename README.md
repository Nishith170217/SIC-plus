# SIC+: Extended Similarity-Based Interpretable Image Classification

This repository extends the official implementation of **SIC: Similarity-Based Interpretable Image Classification with Neural Networks** by Wolf et al. (ICCV 2025).

The project was developed for the Representation Learning course at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU). It reproduces the original Stanford Dogs experiment, adds Pascal VOC support to the released codebase, evaluates SIC on Oxford-IIIT Pet, and studies support selection, robustness, support count, and backbone efficiency.

## Main extensions

| Extension | Description |
| --- | --- |
| Pascal VOC 2007 | Multi-label SIC implementation using B-cos DenseNet121 |
| Oxford-IIIT Pet | Evaluation on a new 37-class fine-grained classification dataset |
| Support count | Comparison of 1, 3, and 5 supports per class on Pascal VOC |
| Support selection | Comparison of k-means, random, maximum-diversity, and compact supports |
| Support diversity | Analysis of the relationship between support diversity and class performance |
| Corruption robustness | Evaluation under Gaussian noise, Gaussian blur, and central occlusion |
| Backbone comparison | B-cos ResNet50 versus B-cos DenseNet121 on Oxford-IIIT Pet |
| SIC visualizations | Qualitative explanations for correct predictions, errors, corruptions, and different supports |

## Results

### Stanford Dogs reproduction

Stanford Dogs is a 120-class single-label classification dataset.

| Metric | Paper | Ours |
| --- | ---: | ---: |
| Top-1 accuracy | 83.46% | 78.41% |
| Top-5 accuracy | — | 96.82% |
| Macro F1 | — | 77.56% |
| Balanced accuracy | — | 77.95% |

### Pascal VOC 2007

Pascal VOC is treated as a 20-class multi-label classification problem. Images may contain multiple object classes, and difficult labels are ignored during loss and metric computation.

| Metric | Result |
| --- | ---: |
| Sklearn mAP | 86.80% |
| VOC 2007 11-point mAP | 83.58% |
| Label accuracy at 0.5 | 97.05% |
| Micro F1 | 76.96% |
| Macro F1 | 81.11% |

Label accuracy, mAP, and F1 measure different aspects of multi-label performance. Label accuracy is included for completeness, but mAP and F1 are more informative for the imbalanced Pascal VOC setting.

### Oxford-IIIT Pet

| Backbone | Top-1 | Top-5 | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| B-cos ResNet50 | **92.89%** | **99.32%** | **92.75%** | **92.83%** |
| B-cos DenseNet121 | 91.77% | 98.94% | 91.68% | 91.71% |

### Number of supports on Pascal VOC

| Supports per class | mAP | VOC07 mAP | Micro F1 | Macro F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 84.35% | 81.90% | 75.14% | 78.98% |
| 3 | **86.80%** | **83.58%** | **76.96%** | **81.11%** |
| 5 | 84.20% | 81.54% | 72.65% | 76.32% |

Three supports per class gave the best overall result. Increasing the support count from three to five did not improve performance.

### Support selection

On Pascal VOC, replacing the standard k-means supports with compact representative supports improved mAP from **86.80% to 88.09%** without retraining the network.

| Strategy | VOC mAP | Change |
| --- | ---: | ---: |
| K-means | 86.80% | — |
| Compact representative | **88.09%** | **+1.29** |
| Maximum diversity | 56.74% | −30.05 |

The compact strategy improved 14 of the 20 VOC classes. The per-class improvement was significant with a paired t-test (`p = 0.015`) and a Wilcoxon signed-rank test (`p = 0.024`).

Maximum diversity performed poorly. This suggests that useful supports should be representative of their class rather than simply being as different from one another as possible.

On Stanford Dogs, compact supports improved Top-1 accuracy from 78.41% to 78.75%, but the per-class improvement was not statistically significant.

### Corruption robustness

The trained models were evaluated without retraining under three corruption types and three severity levels.

#### Pascal VOC mAP

| Corruption | Mild | Medium | Severe |
| --- | ---: | ---: | ---: |
| Gaussian noise | 85.67% | 82.84% | 73.47% |
| Gaussian blur | 85.38% | 80.33% | 62.06% |
| Central occlusion | 83.05% | 70.59% | 45.26% |

Clean VOC mAP was 86.80%.

#### Stanford Dogs Top-1 accuracy

| Corruption | Mild | Medium | Severe |
| --- | ---: | ---: | ---: |
| Gaussian noise | 73.08% | 58.30% | 24.35% |
| Gaussian blur | 74.53% | 59.32% | 30.42% |
| Central occlusion | 63.60% | 41.28% | 18.75% |

Clean Dogs Top-1 accuracy was 78.41%. Central occlusion caused the largest performance reduction on both datasets.

### Backbone comparison

The ResNet50 and DenseNet121 backbones were trained on Oxford-IIIT Pet using the same dataset split, support count, optimizer, learning rate, batch settings, and number of epochs.

| Backbone | Top-1 | Parameters | Training time | Checkpoint size |
| --- | ---: | ---: | ---: | ---: |
| B-cos ResNet50 | **92.89%** | 23.75M | 7 h 06 min | 95.3 MB |
| B-cos DenseNet121 | 91.77% | **7.05M** | **5 h 33 min** | **28.8 MB** |

DenseNet121 reduced Top-1 accuracy by 1.12 percentage points but used approximately 70% fewer parameters, trained about 22% faster, and produced a checkpoint approximately 70% smaller.

## Installation

### Requirements

- Conda
- Python 3.12
- CUDA-capable NVIDIA GPU for training
- SLURM only when using the included HPC job scripts

Create the project environment:

```bash
conda create --name sic-voc python=3.12 -y
conda activate sic-voc

pip install torch==2.6.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install tqdm einops matplotlib omegaconf \
  pandas scikit-learn scipy pillow

pip install -e setup/
```

Check the installation:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import bcos; print('B-cos import passed')"
```

## Datasets

Dataset paths are passed through command-line arguments and are not stored in the repository.

### Stanford Dogs

Download the images and annotation lists from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

Expected source structure:

```text
dogs/
├── Images/
├── train_list.mat
├── test_list.mat
└── file_list.mat
```

Generate the processed CSV files:

```bash
python dogs_dataset.py --data_dir=/path/to/dogs
```

The processed directory should contain:

```text
dogs/
├── Images/
└── processed/
    ├── train.csv
    └── valid.csv
```

### Pascal VOC 2007

Download the VOC 2007 train/validation and test archives and extract both into the same directory.

Expected structure:

```text
VOCdevkit/
└── VOC2007/
    ├── Annotations/
    ├── ImageSets/
    │   └── Main/
    ├── JPEGImages/
    ├── SegmentationClass/
    └── SegmentationObject/
```

Check the loader:

```bash
python voc_dataset.py
```

The expected split sizes are:

- Train/validation: 5,011 images
- Test: 4,952 images
- Classes: 20

### Oxford-IIIT Pet

Download `images.tar.gz` and `annotations.tar.gz` from the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), then extract both archives.

Expected structure:

```text
oxford-iiit-pet/
├── images/
└── annotations/
    ├── trainval.txt
    ├── test.txt
    ├── trimaps/
    └── xmls/
```

Check the loader:

```bash
python pets_dataset.py \
  --data_dir=/path/to/oxford-iiit-pet
```

The expected split sizes are:

- Train/validation: 3,680 images
- Test: 3,669 images
- Classes: 37

## Training

### Stanford Dogs

```bash
python train_sic.py \
  --data_dir=/path/to/dogs/processed \
  --epochs=50 \
  --batch_size=8 \
  --n_way=30 \
  --accumulation_steps=4 \
  --results_dir=results/dogs
```

The corresponding SLURM script is:

```bash
sbatch train_dogs.sh
```

### Pascal VOC

```bash
python train_sic_voc.py \
  --data_dir=/path/to/VOCdevkit/VOC2007 \
  --results_dir=results/voc_full \
  --epochs=50 \
  --batch_size=8 \
  --n_shot=3 \
  --accumulation_steps=2 \
  --num_workers=8 \
  --support_batch_size=32 \
  --val_batch_size=32 \
  --lr=0.0001 \
  --amp
```

The full SLURM job can be submitted with:

```bash
sbatch train_voc_full.sh
```

### Oxford-IIIT Pet with ResNet50

```bash
python train_sic_pets.py \
  --data_dir=/path/to/oxford-iiit-pet \
  --results_dir=results/pets_full \
  --epochs=50 \
  --batch_size=8 \
  --n_way=30 \
  --n_shot=3 \
  --backbone=resnet50 \
  --accumulation_steps=4 \
  --num_workers=8 \
  --support_batch_size=32 \
  --val_batch_size=32 \
  --lr=0.001 \
  --amp
```

### Oxford-IIIT Pet with DenseNet121

Use the same settings and change the backbone:

```bash
python train_sic_pets.py \
  --data_dir=/path/to/oxford-iiit-pet \
  --results_dir=results/pets_densenet_full \
  --epochs=50 \
  --batch_size=8 \
  --n_way=30 \
  --n_shot=3 \
  --backbone=densenet121 \
  --accumulation_steps=4 \
  --num_workers=8 \
  --support_batch_size=32 \
  --val_batch_size=32 \
  --lr=0.001 \
  --amp
```

## Evaluation

### Pascal VOC

```bash
python eval_voc.py \
  --data_dir=/path/to/VOCdevkit/VOC2007 \
  --checkpoint=results/voc_full/best_model.pth \
  --output=results/voc_full/evaluation.json \
  --n_shot=3
```

This reports sklearn mAP, VOC 2007 11-point mAP, label accuracy, Micro F1, Macro F1, and per-class AP.

### Oxford-IIIT Pet

ResNet50:

```bash
python eval_pets.py \
  --data_dir=/path/to/oxford-iiit-pet \
  --checkpoint=results/pets_full/best_model.pth \
  --output=results/pets_full/evaluation.json \
  --backbone=resnet50 \
  --n_shot=3
```

DenseNet121:

```bash
python eval_pets.py \
  --data_dir=/path/to/oxford-iiit-pet \
  --checkpoint=results/pets_densenet_full/best_model.pth \
  --output=results/pets_densenet_full/evaluation.json \
  --backbone=densenet121 \
  --n_shot=3
```

## Visualizations

Generate Pascal VOC explanations:

```bash
python vis_voc_sic.py \
  --data_dir=/path/to/VOCdevkit/VOC2007 \
  --checkpoint=results/voc_full/best_model.pth \
  --output_dir=visualizations/voc \
  --sample_index=6 \
  --n_shot=3
```

Generate Oxford Pets explanations:

```bash
python vis_pets_sic.py \
  --data_dir=/path/to/oxford-iiit-pet \
  --checkpoint=results/pets_full/best_model.pth \
  --output_dir=visualizations/pets \
  --sample_indices 0 10 119 148 \
  --backbone=resnet50 \
  --n_shot=3
```

The visualization output includes:

- query image;
- predicted and target class;
- prediction confidence;
- selected support images;
- support evidence;
- query contribution map;
- support contribution maps.

## Experiment outputs

Compact experiment results are stored under:

```text
experiments/
├── backbone_comparison/
├── corruption_robustness/
├── cross_dataset/
├── nshot/
├── pets/
├── support_diversity/
└── support_selection/
```

Reusable analysis scripts are stored under:

```text
analysis/
├── backbone_comparison/
├── corruption_robustness/
├── cross_dataset/
├── pets/
├── support_diversity/
└── support_selection/
```

Model checkpoints, logs, and generated high-resolution figures are excluded from Git because of their size.

## Main findings

1. SIC was successfully adapted to Pascal VOC multi-label classification.
2. SIC generalized well to Oxford-IIIT Pet, reaching 92.89% Top-1 accuracy.
3. Three supports per class performed better than one or five on Pascal VOC.
4. Compact representative supports improved VOC mAP without retraining.
5. Maximum support diversity did not improve performance and often selected poor representatives.
6. SIC performance deteriorated under severe corruption, especially central occlusion.
7. DenseNet121 offered a strong efficiency–accuracy trade-off compared with ResNet50.
8. SIC visualizations helped identify relevant image evidence and understand high-confidence errors.

## Limitations

- Only two backbone architectures were compared.
- Robustness was evaluated using synthetic corruptions.
- Explanation quality was mainly assessed qualitatively.
- Full repeated training across several random seeds was limited by computational cost.
- Metrics from single-label and multi-label datasets are not directly comparable.
- Support-selection results may depend on the dataset and learned feature space.

## Acknowledgements

This project builds on the official SIC implementation by Tom Nuno Wolf and the authors of:

> SIC: Similarity-Based Interpretable Image Classification with Neural Networks, ICCV 2025.

The original code remains subject to its original Apache 2.0 license. Dataset rights remain with the respective dataset owners.

## License

This repository follows the Apache License 2.0 included in the project.
