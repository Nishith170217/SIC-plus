# Standalone evaluation for SIC on Oxford-IIIT Pet.

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.nn.functional import one_hot
from tqdm import tqdm

from bcos import BcosEncoderWrapper, resnet50_long
from bcos.pretrained_imagenet import densenet121_long
from pets_dataset import N_PET_CLASSES, get_pets_dataloader
from sic import SIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SIC on Oxford-IIIT Pet."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--support_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--backbone",
        choices=["resnet50", "densenet121"],
        default="resnet50",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    if args.support_batch_size < 1:
        parser.error("--support_batch_size must be at least 1")

    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")

    if args.n_shot < 1:
        parser.error("--n_shot must be at least 1")

    return args


def get_class_names(data_dir):
    """Recover the official class-ID to breed-name mapping."""
    class_names = [None] * N_PET_CLASSES
    split_path = data_dir / "annotations" / "trainval.txt"

    with split_path.open("r") as handle:
        for line in handle:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            image_name = parts[0]
            class_index = int(parts[1]) - 1
            breed_name = image_name.rsplit("_", 1)[0]

            class_names[class_index] = breed_name

    for class_index, class_name in enumerate(class_names):
        if class_name is None:
            class_names[class_index] = f"class_{class_index}"

    return class_names


def load_model(args, support_loader, device):
    if args.backbone == "resnet50":
        backbone = resnet50_long(pretrained=False)
    else:
        backbone = densenet121_long(pretrained=False)

    print(f"Backbone: {args.backbone}")

    featurizer = BcosEncoderWrapper(backbone)

    model = SIC(
        featurizer=featurizer,
        n_classes=N_PET_CLASSES,
        proj_dim=128,
        n_way=30,
        n_shot=args.n_shot,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=False,
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate(model, validation_loader, device, class_names):
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    total_images = 0
    top5_correct = 0

    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="sum"
    )

    with torch.no_grad():
        for images, targets in tqdm(
            validation_loader,
            desc="Evaluating Oxford Pets",
        ):
            images = images.to(device)
            targets = targets.to(device).long()

            logits = model.predict(images)

            binary_targets = one_hot(
                targets,
                num_classes=N_PET_CLASSES,
            ).float()

            batch_loss = criterion(
                logits,
                binary_targets,
            )

            predictions = logits.argmax(dim=1)
            top5_predictions = logits.topk(
                k=5,
                dim=1,
            ).indices

            top5_correct += (
                top5_predictions == targets.unsqueeze(1)
            ).any(dim=1).sum().item()

            total_loss += batch_loss.item()
            total_images += targets.size(0)

            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()

    top1_accuracy = (
        np.mean(predictions == targets) * 100.0
    )
    top5_accuracy = (
        top5_correct / total_images * 100.0
    )
    macro_f1 = (
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )
    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions,
        )
        * 100.0
    )

    per_class = {}

    for class_index, class_name in enumerate(class_names):
        class_mask = targets == class_index
        class_count = int(class_mask.sum())

        if class_count == 0:
            class_accuracy = None
        else:
            class_accuracy = float(
                np.mean(
                    predictions[class_mask]
                    == targets[class_mask]
                )
                * 100.0
            )

        per_class[class_name] = {
            "class_index": class_index,
            "test_images": class_count,
            "accuracy": class_accuracy,
        }

    return {
        "dataset": "Oxford-IIIT Pet",
        "test_images": int(total_images),
        "n_classes": N_PET_CLASSES,
        "n_shot": model.n_shot,
        "loss": float(
            total_loss
            / (total_images * N_PET_CLASSES)
        ),
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "per_class": per_class,
    }


def main():
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    _, validation_loader, support_loader = (
        get_pets_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_bcos=True,
            support_batch_size=(
                args.support_batch_size
            ),
            val_batch_size=args.batch_size,
        )
    )

    class_names = get_class_names(args.data_dir)

    print("Loading trained model...")
    model = load_model(
        args,
        support_loader,
        device,
    )

    print("Precomputing support prototypes...")
    with torch.no_grad():
        model.precompute()

    metrics = evaluate(
        model,
        validation_loader,
        device,
        class_names,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print()
    print("Oxford-IIIT Pet evaluation")
    print("=" * 34)
    print(
        f"Test images:       "
        f"{metrics['test_images']}"
    )
    print(
        f"Top-1 accuracy:    "
        f"{metrics['top1_accuracy']:.2f}%"
    )
    print(
        f"Top-5 accuracy:    "
        f"{metrics['top5_accuracy']:.2f}%"
    )
    print(
        f"Macro F1:          "
        f"{metrics['macro_f1']:.2f}%"
    )
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.2f}%"
    )
    print()
    print("Per-class accuracy:")

    for class_name, class_result in metrics[
        "per_class"
    ].items():
        accuracy = class_result["accuracy"]
        count = class_result["test_images"]

        if accuracy is None:
            accuracy_text = "N/A"
        else:
            accuracy_text = f"{accuracy:.2f}%"

        print(
            f"{class_name:30s} "
            f"n={count:3d} "
            f"accuracy={accuracy_text}"
        )

    print(f"\nSaved evaluation to: {args.output}")


if __name__ == "__main__":
    main()
