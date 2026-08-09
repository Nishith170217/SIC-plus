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
from torch.utils.data import DataLoader
from tqdm import tqdm

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from dogs_dataset import EvalTransform
from eval_voc import compute_metrics
from sic import SIC
from voc_dataset import VOC_CLASSES, VOCDataset, get_voc_dataloader


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


class CorruptedVOCDataset(VOCDataset):
    def __init__(
        self,
        root,
        corruption,
        severity,
        transform,
        seed=42,
    ):
        super().__init__(
            root,
            split="test",
            transform=transform,
        )
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
                loc=0.0,
                scale=self.severity,
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
        image_id = self.image_ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.corrupt(image, index)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen SIC on corrupted VOC2007 "
            "test images."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--support_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}"
        )

    return args


def build_model(
    support_loader,
    checkpoint,
    device,
):
    featurizer = BcosEncoderWrapper(
        densenet121_long(pretrained=False)
    )

    model = SIC(
        featurizer=featurizer,
        n_classes=len(VOC_CLASSES),
        proj_dim=128,
        n_way=None,
        n_shot=3,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=True,
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


def evaluate(
    model,
    loader,
    device,
    threshold,
):
    all_scores = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in tqdm(loader):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model.predict(images)
            scores = torch.sigmoid(logits)

            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.numpy())

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)

    return compute_metrics(
        scores,
        targets,
        threshold=threshold,
    )


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

    _, _, support_loader = get_voc_dataloader(
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

        dataset = CorruptedVOCDataset(
            args.data_dir,
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

        metrics = evaluate(
            model,
            loader,
            device,
            args.threshold,
        )

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
            f"mAP={metrics['sklearn_map']:.2f}%, "
            f"accuracy="
            f"{metrics['label_accuracy_at_threshold']:.2f}%, "
            f"micro_F1={metrics['micro_f1']:.2f}%, "
            f"macro_F1={metrics['macro_f1']:.2f}%"
        )

    clean = results[0]
    comparison_rows = []

    for result in results:
        comparison_rows.append(
            {
                "corruption": result["corruption"],
                "severity": result["severity"],
                "sklearn_map": result["sklearn_map"],
                "delta_sklearn_map": (
                    result["sklearn_map"]
                    - clean["sklearn_map"]
                ),
                "voc07_map": (
                    result["voc_2007_11_point_map"]
                ),
                "delta_voc07_map": (
                    result["voc_2007_11_point_map"]
                    - clean["voc_2007_11_point_map"]
                ),
                "label_accuracy": (
                    result[
                        "label_accuracy_at_threshold"
                    ]
                ),
                "delta_label_accuracy": (
                    result[
                        "label_accuracy_at_threshold"
                    ]
                    - clean[
                        "label_accuracy_at_threshold"
                    ]
                ),
                "micro_f1": result["micro_f1"],
                "delta_micro_f1": (
                    result["micro_f1"]
                    - clean["micro_f1"]
                ),
                "macro_f1": result["macro_f1"],
                "delta_macro_f1": (
                    result["macro_f1"]
                    - clean["macro_f1"]
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
        "dataset": "Pascal VOC 2007 test",
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

    print("\nVOC corruption evaluation complete")
    print(f"Comparison: {comparison_path}")
    print(
        f"Summary: {args.output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
