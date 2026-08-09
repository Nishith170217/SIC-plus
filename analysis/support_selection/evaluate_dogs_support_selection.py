import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from bcos import BcosEncoderWrapper, resnet50_long
from dogs_dataset import get_dogs_dataloader
from nw.utils import compute_clusters
from sic import SIC


N_CLASSES = 120


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SIC support-selection strategies "
            "on Stanford Dogs."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--support_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if args.n_shot < 1:
        parser.error("--n_shot must be at least 1")

    return args


def build_model(support_loader, checkpoint, device, n_shot):
    featurizer = BcosEncoderWrapper(
        resnet50_long(pretrained=False)
    )

    model = SIC(
        featurizer=featurizer,
        n_classes=N_CLASSES,
        proj_dim=128,
        n_way=30,
        n_shot=n_shot,
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

    return model


def extract_support_features(model, support_loader, device):
    features = []
    targets = []

    print("Extracting Dogs support features...")

    with torch.inference_mode():
        for images, labels in tqdm(support_loader):
            images = images.to(device, non_blocking=True)
            batch_features = model.featurizer(images)

            features.append(batch_features.detach().cpu())
            targets.append(labels.detach().cpu().long())

    return torch.cat(features), torch.cat(targets)


def representative_first_indices(
    class_features,
    n_shot,
    maximize_diversity,
):
    normalized = functional.normalize(
        class_features,
        dim=1,
    )

    centroid = functional.normalize(
        normalized.mean(dim=0, keepdim=True),
        dim=1,
    )

    centroid_similarity = (
        normalized @ centroid.transpose(0, 1)
    ).squeeze(1)

    selected = [int(centroid_similarity.argmax())]

    while len(selected) < n_shot:
        candidates = [
            index
            for index in range(len(class_features))
            if index not in selected
        ]

        similarities = (
            normalized[candidates]
            @ normalized[selected].transpose(0, 1)
        )

        if maximize_diversity:
            minimum_distances = (
                1.0 - similarities
            ).min(dim=1).values
            position = int(minimum_distances.argmax())
        else:
            mean_similarities = similarities.mean(dim=1)
            position = int(mean_similarities.argmax())

        selected.append(candidates[position])

    return torch.tensor(selected, dtype=torch.long)


def select_supports(
    features,
    targets,
    n_shot,
    strategy,
    seed=None,
):
    selected_features = []
    selected_labels = []
    selected_indices = []

    generator = torch.Generator()

    if seed is not None:
        generator.manual_seed(seed)

    for class_index in range(N_CLASSES):
        class_indices = torch.where(
            targets == class_index
        )[0]
        class_features = features[class_indices]

        if len(class_indices) < n_shot:
            raise RuntimeError(
                f"Class {class_index} has only "
                f"{len(class_indices)} samples"
            )

        if strategy == "random":
            local_indices = torch.randperm(
                len(class_indices),
                generator=generator,
            )[:n_shot]

        elif strategy == "max_diversity":
            local_indices = representative_first_indices(
                class_features,
                n_shot,
                maximize_diversity=True,
            )

        elif strategy == "max_similarity":
            local_indices = representative_first_indices(
                class_features,
                n_shot,
                maximize_diversity=False,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        dataset_indices = class_indices[local_indices]

        selected_features.append(
            features[dataset_indices]
        )
        selected_labels.extend(
            [class_index] * n_shot
        )
        selected_indices.extend(
            dataset_indices.tolist()
        )

    return (
        torch.cat(selected_features),
        torch.tensor(selected_labels, dtype=torch.long),
        torch.tensor(selected_indices, dtype=torch.long),
    )


def mean_support_diversity(features, n_shot):
    class_diversities = []

    for class_index in range(N_CLASSES):
        start = class_index * n_shot
        end = start + n_shot

        normalized = functional.normalize(
            features[start:end],
            dim=1,
        )
        similarity = normalized @ normalized.transpose(0, 1)

        upper = torch.triu_indices(
            n_shot,
            n_shot,
            offset=1,
        )

        pairwise = similarity[
            upper[0],
            upper[1],
        ]

        class_diversities.append(
            1.0 - float(pairwise.mean())
        )

    return float(np.mean(class_diversities))


def get_class_names(dataset):
    names = []

    targets = np.asarray(dataset.targets)

    for class_index in range(N_CLASSES):
        first_index = int(
            np.where(targets == class_index)[0][0]
        )
        relative_path = Path(
            dataset.file_paths[first_index]
        )
        names.append(relative_path.parent.name)

    return names


def evaluate(model, val_loader, device, class_names):
    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in tqdm(val_loader):
            images = images.to(device, non_blocking=True)
            logits = model.predict(images)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu().long())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    predictions = logits.argmax(dim=1)
    top_five = logits.topk(
        k=5,
        dim=1,
    ).indices

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

    per_class = {}

    for class_index, class_name in enumerate(class_names):
        mask = targets == class_index
        count = int(mask.sum())
        correct = int(
            (predictions[mask] == class_index).sum()
        )

        per_class[class_name] = {
            "class_index": class_index,
            "sample_count": count,
            "correct_count": correct,
            "accuracy": (
                100.0 * correct / count
                if count > 0
                else float("nan")
            ),
        }

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
    }


def install_supports(
    model,
    features,
    labels,
    indices,
):
    model.cluster_feat = features
    model.cluster_y = labels
    model.cluster_indices = indices


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    _, val_loader, support_loader = get_dogs_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=args.batch_size,
    )

    support_dataset = support_loader.dataset
    class_names = get_class_names(support_dataset)

    model = build_model(
        support_loader,
        args.checkpoint,
        device,
        args.n_shot,
    )

    features, targets = extract_support_features(
        model,
        support_loader,
        device,
    )

    configurations = []

    print("Constructing K-means baseline...")
    baseline_features, baseline_labels, baseline_indices = (
        compute_clusters(
            features,
            targets,
            args.n_shot,
        )
    )

    configurations.append(
        (
            "kmeans",
            "kmeans",
            None,
            baseline_features,
            baseline_labels,
            baseline_indices,
        )
    )

    for seed in args.random_seeds:
        selected = select_supports(
            features,
            targets,
            args.n_shot,
            strategy="random",
            seed=seed,
        )

        configurations.append(
            (
                f"random_seed_{seed}",
                "random",
                seed,
                *selected,
            )
        )

    for strategy in ["max_diversity", "max_similarity"]:
        selected = select_supports(
            features,
            targets,
            args.n_shot,
            strategy=strategy,
        )

        configurations.append(
            (
                strategy,
                strategy,
                None,
                *selected,
            )
        )

    results = []

    for (
        configuration_name,
        strategy,
        seed,
        selected_features,
        selected_labels,
        selected_indices,
    ) in configurations:
        print(f"\nEvaluating {configuration_name}...")

        install_supports(
            model,
            selected_features,
            selected_labels,
            selected_indices,
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
            class_names,
        )

        diversity = mean_support_diversity(
            selected_features,
            args.n_shot,
        )

        selected_paths = [
            support_dataset.file_paths[index]
            for index in selected_indices.tolist()
        ]

        result = {
            "configuration": configuration_name,
            "strategy": strategy,
            "seed": seed,
            "n_shot": args.n_shot,
            "mean_support_diversity": diversity,
            "selected_indices": selected_indices.tolist(),
            "selected_paths": selected_paths,
            **metrics,
        }

        save_json(
            result,
            args.output_dir
            / f"{configuration_name}.json",
        )

        results.append(result)

        print(
            f"{configuration_name}: "
            f"diversity={diversity:.4f}, "
            f"top1={metrics['top1_accuracy']:.2f}%, "
            f"top5={metrics['top5_accuracy']:.2f}%, "
            f"macro_F1={metrics['macro_f1']:.2f}%"
        )

    baseline = results[0]
    comparison_rows = []

    for result in results:
        comparison_rows.append(
            {
                "configuration": result["configuration"],
                "strategy": result["strategy"],
                "seed": result["seed"],
                "mean_support_diversity": (
                    result["mean_support_diversity"]
                ),
                "top1_accuracy": result["top1_accuracy"],
                "delta_top1": (
                    result["top1_accuracy"]
                    - baseline["top1_accuracy"]
                ),
                "top5_accuracy": result["top5_accuracy"],
                "delta_top5": (
                    result["top5_accuracy"]
                    - baseline["top5_accuracy"]
                ),
                "macro_f1": result["macro_f1"],
                "delta_macro_f1": (
                    result["macro_f1"]
                    - baseline["macro_f1"]
                ),
                "balanced_accuracy": (
                    result["balanced_accuracy"]
                ),
                "delta_balanced_accuracy": (
                    result["balanced_accuracy"]
                    - baseline["balanced_accuracy"]
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

    random_results = [
        result
        for result in results
        if result["strategy"] == "random"
    ]

    summary = {
        "dataset": "Stanford Dogs",
        "checkpoint": str(args.checkpoint),
        "n_classes": N_CLASSES,
        "n_shot": args.n_shot,
        "baseline": {
            "mean_support_diversity": baseline[
                "mean_support_diversity"
            ],
            "top1_accuracy": baseline["top1_accuracy"],
            "top5_accuracy": baseline["top5_accuracy"],
            "macro_f1": baseline["macro_f1"],
            "balanced_accuracy": baseline[
                "balanced_accuracy"
            ],
        },
        "random_mean": {
            "top1_accuracy": float(
                np.mean(
                    [
                        result["top1_accuracy"]
                        for result in random_results
                    ]
                )
            ),
            "top1_std": float(
                np.std(
                    [
                        result["top1_accuracy"]
                        for result in random_results
                    ],
                    ddof=1,
                )
            ),
            "macro_f1": float(
                np.mean(
                    [
                        result["macro_f1"]
                        for result in random_results
                    ]
                )
            ),
            "macro_f1_std": float(
                np.std(
                    [
                        result["macro_f1"]
                        for result in random_results
                    ],
                    ddof=1,
                )
            ),
        },
        "configurations": comparison_rows,
    }

    save_json(
        summary,
        args.output_dir / "summary.json",
    )

    print("\nDogs support-selection experiment complete")
    print(f"Comparison: {comparison_path}")
    print(
        f"Summary: {args.output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
