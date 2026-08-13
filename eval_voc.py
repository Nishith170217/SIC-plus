# Evaluate trained SIC on Pascal VOC 2007.

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from sic import SIC
from voc_dataset import VOC_CLASSES, get_voc_dataloader


def voc_2007_ap(targets, scores):
    """VOC 2007 11-point interpolated average precision."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]

    true_positives = np.cumsum(sorted_targets == 1)
    false_positives = np.cumsum(sorted_targets == 0)

    precision = true_positives / np.maximum(
        true_positives + false_positives,
        1,
    )
    recall = true_positives / np.maximum(
        (targets == 1).sum(),
        1,
    )

    ap = 0.0

    for recall_threshold in np.arange(0.0, 1.1, 0.1):
        eligible = recall >= recall_threshold
        interpolated_precision = (
            precision[eligible].max()
            if eligible.any()
            else 0.0
        )
        ap += interpolated_precision / 11.0

    return float(ap)


def compute_metrics(scores, targets, threshold=0.5):
    if scores.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: {scores.shape} versus {targets.shape}"
        )

    sklearn_aps = []
    voc_2007_aps = []
    per_class = {}

    for class_idx, class_name in enumerate(VOC_CLASSES):
        class_targets = targets[:, class_idx]
        class_scores = scores[:, class_idx]

        # Ignore VOC difficult labels encoded as -1.
        valid = class_targets >= 0
        class_targets = class_targets[valid].astype(np.int64)
        class_scores = class_scores[valid]

        positive_count = int((class_targets == 1).sum())
        negative_count = int((class_targets == 0).sum())

        if positive_count == 0:
            sklearn_ap = np.nan
            voc_ap = np.nan
        else:
            sklearn_ap = average_precision_score(
                class_targets,
                class_scores,
            )
            voc_ap = voc_2007_ap(
                class_targets,
                class_scores,
            )

        sklearn_aps.append(sklearn_ap)
        voc_2007_aps.append(voc_ap)

        per_class[class_name] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sklearn_ap": float(sklearn_ap * 100),
            "voc_2007_11_point_ap": float(voc_ap * 100),
        }

    sklearn_map = float(
        np.nanmean(sklearn_aps) * 100
    )
    voc_2007_map = float(
        np.nanmean(voc_2007_aps) * 100
    )

    predictions = (scores >= threshold).astype(np.int64)
    valid = targets >= 0
    valid_targets = targets[valid].astype(np.int64)
    valid_predictions = predictions[valid]

    label_accuracy = float(
        (valid_targets == valid_predictions).mean() * 100
    )
    micro_f1 = float(
        f1_score(
            valid_targets,
            valid_predictions,
            average="binary",
            zero_division=0,
        )
        * 100
    )

    class_f1_scores = []

    for class_idx in range(targets.shape[1]):
        class_valid = targets[:, class_idx] >= 0
        class_targets = targets[
            class_valid,
            class_idx,
        ].astype(np.int64)
        class_predictions = predictions[
            class_valid,
            class_idx,
        ]

        class_f1_scores.append(
            f1_score(
                class_targets,
                class_predictions,
                average="binary",
                zero_division=0,
            )
        )

    macro_f1 = float(
        np.mean(class_f1_scores) * 100
    )

    return {
        "sklearn_map": sklearn_map,
        "voc_2007_11_point_map": voc_2007_map,
        "label_accuracy_at_threshold": label_accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "threshold": threshold,
        "per_class": per_class,
    }


def parse_args():
    parser = argparse.ArgumentParser()

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
        default=Path("results/voc_full/evaluation.json"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--n_shot",
        type=int,
        default=3,
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
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if args.n_shot < 1:
        parser.error("--n_shot must be at least 1")

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}"
        )

    return args


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    _, val_loader, support_loader = get_voc_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=args.batch_size,
    )

    backbone = densenet121_long(pretrained=False)
    featurizer = BcosEncoderWrapper(backbone)

    model = SIC(
        featurizer=featurizer,
        n_classes=len(VOC_CLASSES),
        proj_dim=128,
        n_way=None,
        n_shot=args.n_shot,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=True,
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("Precomputing support vectors...")
    model.precompute()

    all_scores = []
    all_targets = []

    print("Evaluating complete VOC2007 test split...")

    with torch.inference_mode():
        for images, targets in tqdm(val_loader):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model.predict(images)
            scores = torch.sigmoid(logits)

            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(
        all_scores,
        all_targets,
        threshold=args.threshold,
    )

    print("\nEvaluation results")
    print("=" * 60)
    print(
        f"Sklearn mAP:              "
        f"{metrics['sklearn_map']:.2f}%"
    )
    print(
        f"VOC2007 11-point mAP:     "
        f"{metrics['voc_2007_11_point_map']:.2f}%"
    )
    print(
        f"Label accuracy @ {args.threshold:.2f}: "
        f"{metrics['label_accuracy_at_threshold']:.2f}%"
    )
    print(
        f"Micro F1:                 "
        f"{metrics['micro_f1']:.2f}%"
    )
    print(
        f"Macro F1:                 "
        f"{metrics['macro_f1']:.2f}%"
    )

    print("\nPer-class AP")
    print("=" * 60)

    for class_name in VOC_CLASSES:
        values = metrics["per_class"][class_name]
        print(
            f"{class_name:12s} "
            f"positive={values['positive_count']:4d} "
            f"sklearn={values['sklearn_ap']:6.2f}% "
            f"VOC07={values['voc_2007_11_point_ap']:6.2f}%"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.output, "w") as output_file:
        json.dump(
            metrics,
            output_file,
            indent=2,
        )

    print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
