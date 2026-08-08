# Pascal VOC 2007 dataset loader for SIC
# Multi-label classification with 20 classes

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import Tensor

from dogs_dataset import TrainTransform, EvalTransform

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):

    def __init__(self, root, split='trainval', transform=None):
        self.root = Path(root)
        self.image_dir = self.root / 'JPEGImages'
        self.annotation_dir = self.root / 'ImageSets' / 'Main'
        self.transform = transform
        self.split = split

        # Load image IDs for this split
        split_file = self.annotation_dir / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        # Build label matrix — shape (N, 20)
        n = len(self.image_ids)
        self.labels = torch.zeros(n, len(VOC_CLASSES), dtype=torch.float32)
       # Compatibility with SIC support-set code
        self.targets = self.labels
        id_to_idx = {img_id: idx for idx, img_id in enumerate(self.image_ids)}

        for cls_idx, cls_name in enumerate(VOC_CLASSES):
            cls_file = self.annotation_dir / f'{cls_name}_{split}.txt'
            if not cls_file.exists():
                continue
            with open(cls_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    img_id, label = parts[0], int(parts[1])

                    if label not in (-1, 0, 1):
                        raise ValueError(
                            f"Unexpected VOC label {label} in {cls_file}"
                        )

                    if img_id not in id_to_idx:
                        continue

                    dataset_idx = id_to_idx[img_id]

                    if label == 1:
                    # Positive class
                        self.labels[dataset_idx, cls_idx] = 1.0
                    elif label == 0:
                    # Difficult class: use -1 as the internal ignore value
                        self.labels[dataset_idx, cls_idx] = -1.0
                    # Official -1 remains internal 0: ordinary negative class

        self.targets = self.labels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_dir / f'{img_id}.jpg'
        with Image.open(img_path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_voc_dataloader(data_dir, batch_size, num_workers, is_bcos=True,
                       support_batch_size=None, val_batch_size=None):
    data_dir = Path(data_dir)
    generator = torch.Generator().manual_seed(42)

    train_data = VOCDataset(
        data_dir, split='trainval',
        transform=TrainTransform(is_bcos=is_bcos)
    )
    val_data = VOCDataset(
        data_dir, split='test',
        transform=EvalTransform(is_bcos=is_bcos)
    )
    support_data = VOCDataset(
        data_dir, split='trainval',
        transform=EvalTransform(is_bcos=is_bcos)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size if val_batch_size else batch_size,
        num_workers=num_workers,
    )
    support_loader = DataLoader(
        support_data,
        batch_size=support_batch_size if support_batch_size else batch_size,
        num_workers=num_workers,
    )

    return train_loader, val_loader, support_loader


if __name__ == "__main__":
    # Quick sanity check
    from pathlib import Path
    data_dir = Path('/home/woody/rlvl/rlvl171v/data/pascal-voc/VOCdevkit/VOC2007')
    train_loader, val_loader, support_loader = get_voc_dataloader(
        data_dir, batch_size=4, num_workers=2, is_bcos=True
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    x, y = next(iter(train_loader))
    print(f"Image shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Sample labels: {y[0]}")
    positive_indices = torch.nonzero(y[0] > 0, as_tuple=True)[0].tolist()
    print(f"Classes present: {[VOC_CLASSES[i] for i in positive_indices]}")
