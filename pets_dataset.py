# Oxford-IIIT Pet dataset loader for SIC.
# 37-class single-label breed classification.

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dogs_dataset import EvalTransform, TrainTransform


N_PET_CLASSES = 37


class OxfordPetsDataset(Dataset):
    def __init__(
        self,
        root,
        split="trainval",
        transform=None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.annotation_dir = self.root / "annotations"
        self.transform = transform
        self.split = split

        split_path = (
            self.annotation_dir / f"{split}.txt"
        )

        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        self.image_names = []
        self.targets = []
        self.species = []
        self.breed_ids = []

        with open(split_path) as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) != 4:
                    raise ValueError(
                        f"Malformed line in {split_path}: "
                        f"{line}"
                    )

                image_name = parts[0]
                class_index = int(parts[1]) - 1
                species_index = int(parts[2]) - 1
                breed_index = int(parts[3]) - 1

                if not 0 <= class_index < N_PET_CLASSES:
                    raise ValueError(
                        f"Invalid class index "
                        f"{class_index} for {image_name}"
                    )

                image_path = (
                    self.image_dir / f"{image_name}.jpg"
                )

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                self.image_names.append(image_name)
                self.targets.append(class_index)
                self.species.append(species_index)
                self.breed_ids.append(breed_index)

        self.class_names = self._build_class_names()

    def _build_class_names(self):
        names = [None] * N_PET_CLASSES

        for image_name, target in zip(
            self.image_names,
            self.targets,
        ):
            class_name = image_name.rsplit("_", 1)[0]

            if names[target] is None:
                names[target] = class_name
            elif names[target] != class_name:
                raise ValueError(
                    f"Inconsistent class name for "
                    f"class {target}: {names[target]} "
                    f"versus {class_name}"
                )

        if any(name is None for name in names):
            missing = [
                index
                for index, name in enumerate(names)
                if name is None
            ]
            raise ValueError(
                f"Missing class names for: {missing}"
            )

        return names

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = (
            self.image_dir / f"{image_name}.jpg"
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.targets[index]


def get_pets_dataloader(
    data_dir,
    batch_size,
    num_workers,
    is_bcos=True,
    support_batch_size=None,
    val_batch_size=None,
):
    data_dir = Path(data_dir)
    generator = torch.Generator().manual_seed(42)

    train_data = OxfordPetsDataset(
        data_dir,
        split="trainval",
        transform=TrainTransform(
            random_erase_prob=0.5,
            is_bcos=is_bcos,
        ),
    )

    validation_data = OxfordPetsDataset(
        data_dir,
        split="test",
        transform=EvalTransform(is_bcos=is_bcos),
    )

    support_data = OxfordPetsDataset(
        data_dir,
        split="trainval",
        transform=EvalTransform(is_bcos=is_bcos),
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

    validation_loader = DataLoader(
        validation_data,
        batch_size=(
            val_batch_size
            if val_batch_size
            else batch_size
        ),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    support_loader = DataLoader(
        support_data,
        batch_size=(
            support_batch_size
            if support_batch_size
            else batch_size
        ),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        support_loader,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    args = parser.parse_args()

    train_loader, validation_loader, support_loader = (
        get_pets_dataloader(
            args.data_dir,
            batch_size=4,
            num_workers=2,
            is_bcos=True,
            support_batch_size=4,
            val_batch_size=4,
        )
    )

    images, targets = next(iter(train_loader))

    print(f"Training images: {len(train_loader.dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_loader.dataset)}"
    )
    print(
        f"Support images: "
        f"{len(support_loader.dataset)}"
    )
    print(f"Image shape: {images.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Target dtype: {targets.dtype}")
    print(
        f"Target range: "
        f"{min(train_loader.dataset.targets)} to "
        f"{max(train_loader.dataset.targets)}"
    )
    print(
        f"Classes: "
        f"{len(train_loader.dataset.class_names)}"
    )
    print(
        "First classes:",
        train_loader.dataset.class_names[:5],
    )


if __name__ == "__main__":
    main()
