import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from bcos import BcosEncoderWrapper, resnet50_long
from dogs_dataset import DOGS, EvalTransform, get_dogs_dataloader
from sic import SIC


N_CLASSES = 120

CORRUPTION_CONFIGURATIONS = [
    ("clean", 0.0),
    ("gaussian_noise", 0.05),
    ("gaussian_noise", 0.10),
    ("gaussian_noise", 0.20),
    ("gaussian_blur", 1.0),
    ("gaussian_blur", 2.0),
    ("gaussian_blur", 4.0),
    ("central_occlusion", 0.10),
    ("central_occlusion", 0.25),
    ("central_occlusion", 0.40),
]


class CorruptedDogsDataset(DOGS):
    def __init__(
        self,
        path,
        corruption,
        severity,
        transform,
        seed=42,
    ):
        super().__init__(path, transform)
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def corrupt(self, image, index):
        if self.corruption == "clean":
            return image

        if self.corruption == "gaussian_noise":
            array = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

            generator = np.random.default_rng(
                self.seed + index
            )
            noise = generator.normal(
                0.0,
                self.severity,
                size=array.shape,
            ).astype(np.float32)

            array = np.clip(array + noise, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)

            return Image.fromarray(array, mode="RGB")

        if self.corruption == "gaussian_blur":
            return image.filter(
                ImageFilter.GaussianBlur(
                    radius=self.severity
                )
            )

        if self.corruption == "central_occlusion":
            image = image.copy()
            width, height = image.size

            side_fraction = np.sqrt(self.severity)
            occlusion_width = max(
                1,
                int(width * side_fraction),
            )
            occlusion_height = max(
                1,
                int(height * side_fraction),
            )

            left = (width - occlusion_width) // 2
            top = (height - occlusion_height) // 2
            right = left + occlusion_width
            bottom = top + occlusion_height

            drawer = ImageDraw.Draw(image)
            drawer.rectangle(
                [left, top, right, bottom],
                fill=(127, 127, 127),
            )

            return image

        raise ValueError(
            f"Unknown corruption: {self.corruption}"
        )

    def __getitem__(self, index):
        image_path = self.basepath / self.file_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.corrupt(image, index)

        image = self.transform(image)
        target = self.targets[index]

        return image, target


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen SIC on corrupted Stanford "
            "Dogs validation images."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--support_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}"
        )

    return args


def build_model(support_loader, checkpoint, device):
    featurizer = BcosEncoderWrapper(
        resnet50_long(pretrained=False)
    )

    model = SIC(
        featurizer=featurizer,
        n_classes=N_CLASSES,
        proj_dim=128,
        n_way=30,
        n_shot=3,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=False,
    )

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("Precomputing clean K-means supports...")

    with torch.no_grad():
        model.precompute()

    return model


def evaluate(model, loader, device):
    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in tqdm(loader):
            images = images.to(
                device,
                non_blocking=True,
            )
            logits = model.predict(images)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu().long())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    predictions = logits.argmax(dim=1)
    top_five = logits.topk(k=5, dim=1).indices

    top1_accuracy = float(
        (predictions == targets).float().mean() * 100
    )
    top5_accuracy = float(
        (top_five == targets.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        * 100
    )

    targets_numpy = targets.numpy()
    predictions_numpy = predictions.numpy()

    macro_f1 = float(
        f1_score(
            targets_numpy,
            predictions_numpy,
            average="macro",
            zero_division=0,
        )
        * 100
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            targets_numpy,
            predictions_numpy,
        )
        * 100
    )

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def filename_for(corruption, severity):
    if corruption == "clean":
        return "clean.json"

    severity_text = str(severity).replace(".", "_")
    return f"{corruption}_{severity_text}.json"


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    _, _, support_loader = get_dogs_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=args.batch_size,
    )

    model = build_model(
        support_loader,
        args.checkpoint,
        device,
    )

    transform = EvalTransform(is_bcos=True)
    results = []

    for corruption, severity in CORRUPTION_CONFIGURATIONS:
        print(
            f"\nEvaluating {corruption}, "
            f"severity={severity}"
        )

        dataset = CorruptedDogsDataset(
            args.data_dir / "valid.csv",
            corruption=corruption,
            severity=severity,
            transform=transform,
            seed=args.seed,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        metrics = evaluate(model, loader, device)

        result = {
            "corruption": corruption,
            "severity": severity,
            "query_images_corrupted": (
                corruption != "clean"
            ),
            "support_images_corrupted": False,
            "checkpoint_frozen": True,
            **metrics,
        }

        results.append(result)

        save_json(
            result,
            args.output_dir
            / filename_for(corruption, severity),
        )

        print(
            f"top1={metrics['top1_accuracy']:.2f}%, "
            f"top5={metrics['top5_accuracy']:.2f}%, "
            f"macro_F1={metrics['macro_f1']:.2f}%"
        )

    clean = results[0]
    comparison_rows = []

    for result in results:
        comparison_rows.append(
            {
                "corruption": result["corruption"],
                "severity": result["severity"],
                "top1_accuracy": result["top1_accuracy"],
                "delta_top1": (
                    result["top1_accuracy"]
                    - clean["top1_accuracy"]
                ),
                "top5_accuracy": result["top5_accuracy"],
                "delta_top5": (
                    result["top5_accuracy"]
                    - clean["top5_accuracy"]
                ),
                "macro_f1": result["macro_f1"],
                "delta_macro_f1": (
                    result["macro_f1"]
                    - clean["macro_f1"]
                ),
                "balanced_accuracy": (
                    result["balanced_accuracy"]
                ),
                "delta_balanced_accuracy": (
                    result["balanced_accuracy"]
                    - clean["balanced_accuracy"]
                ),
            }
        )

    comparison_path = args.output_dir / "comparison.csv"

    with open(comparison_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "dataset": "Stanford Dogs validation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_frozen": True,
        "support_configuration": (
            "clean K-means supports, n_shot=3"
        ),
        "corruption_scope": "query images only",
        "seed": args.seed,
        "configurations": comparison_rows,
    }

    save_json(
        summary,
        args.output_dir / "summary.json",
    )

    print("\nDogs corruption evaluation complete")
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
