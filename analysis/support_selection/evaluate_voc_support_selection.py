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
from tqdm import tqdm

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from eval_voc import compute_metrics
from nw.utils import compute_multilabel_clusters
from sic import SIC
from voc_dataset import VOC_CLASSES, get_voc_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate alternative SIC support-selection strategies "
            "on Pascal VOC 2007."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--support_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
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
    backbone = densenet121_long(pretrained=False)
    featurizer = BcosEncoderWrapper(backbone)

    model = SIC(
        featurizer=featurizer,
        n_classes=len(VOC_CLASSES),
        proj_dim=128,
        n_way=None,
        n_shot=n_shot,
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

    return model


def extract_support_features(model, support_loader, device):
    features = []
    targets = []

    print("Extracting trainval support features...")

    with torch.inference_mode():
        for images, labels in tqdm(support_loader):
            images = images.to(device, non_blocking=True)
            batch_features = model.featurizer(images)

            features.append(batch_features.detach().cpu())
            targets.append(labels.detach().cpu())

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

    first_index = int(centroid_similarity.argmax())
    selected = [first_index]

    while len(selected) < n_shot:
        candidates = [
            index
            for index in range(len(class_features))
            if index not in selected
        ]

        candidate_features = normalized[candidates]
        selected_features = normalized[selected]

        similarities = (
            candidate_features
            @ selected_features.transpose(0, 1)
        )

        if maximize_diversity:
            # Select the candidate whose closest selected support
            # is still maximally distant.
            minimum_distances = (
                1.0 - similarities
            ).min(dim=1).values
            chosen_position = int(
                minimum_distances.argmax()
            )
        else:
            # Deliberately select the candidate most similar to
            # the supports already selected.
            mean_similarities = similarities.mean(dim=1)
            chosen_position = int(
                mean_similarities.argmax()
            )

        selected.append(candidates[chosen_position])

    return torch.tensor(selected, dtype=torch.long)


def select_class_supports(
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

    for class_index in range(targets.shape[1]):
        class_indices = torch.where(
            targets[:, class_index] > 0
        )[0]
        class_features = features[class_indices]

        if len(class_indices) < n_shot:
            raise RuntimeError(
                f"{VOC_CLASSES[class_index]} has only "
                f"{len(class_indices)} positive supports"
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


def calculate_mean_diversity(features, n_shot):
    class_diversities = []

    for class_index in range(len(VOC_CLASSES)):
        start = class_index * n_shot
        end = start + n_shot

        class_features = functional.normalize(
            features[start:end],
            dim=1,
        )

        similarities = (
            class_features @ class_features.transpose(0, 1)
        )

        upper_triangle = torch.triu_indices(
            n_shot,
            n_shot,
            offset=1,
        )

        pairwise = similarities[
            upper_triangle[0],
            upper_triangle[1],
        ]

        class_diversities.append(
            1.0 - float(pairwise.mean())
        )

    return float(np.mean(class_diversities))


def evaluate_configuration(
    model,
    val_loader,
    device,
    threshold,
):
    all_scores = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in tqdm(val_loader):
            images = images.to(device, non_blocking=True)
            logits = model.predict(images)
            scores = torch.sigmoid(logits)

            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.numpy())

    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)

    return compute_metrics(
        all_scores,
        all_targets,
        threshold=threshold,
    )


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

    _, val_loader, support_loader = get_voc_dataloader(
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
        args.n_shot,
    )

    features, targets = extract_support_features(
        model,
        support_loader,
        device,
    )

    configurations = []

    print("Constructing K-means baseline supports...")
    kmeans_features, kmeans_labels, kmeans_indices = (
        compute_multilabel_clusters(
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
            kmeans_features,
            kmeans_labels,
            kmeans_indices,
        )
    )

    for seed in args.random_seeds:
        random_features, random_labels, random_indices = (
            select_class_supports(
                features,
                targets,
                args.n_shot,
                strategy="random",
                seed=seed,
            )
        )

        configurations.append(
            (
                f"random_seed_{seed}",
                "random",
                seed,
                random_features,
                random_labels,
                random_indices,
            )
        )

    for strategy in ["max_diversity", "max_similarity"]:
        strategy_features, strategy_labels, strategy_indices = (
            select_class_supports(
                features,
                targets,
                args.n_shot,
                strategy=strategy,
            )
        )

        configurations.append(
            (
                strategy,
                strategy,
                None,
                strategy_features,
                strategy_labels,
                strategy_indices,
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
        print(
            f"\nEvaluating configuration: "
            f"{configuration_name}"
        )

        install_supports(
            model,
            selected_features,
            selected_labels,
            selected_indices,
        )

        metrics = evaluate_configuration(
            model,
            val_loader,
            device,
            args.threshold,
        )

        mean_diversity = calculate_mean_diversity(
            selected_features,
            args.n_shot,
        )

        result = {
            "configuration": configuration_name,
            "strategy": strategy,
            "seed": seed,
            "n_shot": args.n_shot,
            "mean_support_diversity": mean_diversity,
            "selected_indices": selected_indices.tolist(),
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
            f"diversity={mean_diversity:.4f}, "
            f"mAP={metrics['sklearn_map']:.2f}%, "
            f"micro_F1={metrics['micro_f1']:.2f}%, "
            f"macro_F1={metrics['macro_f1']:.2f}%"
        )

    baseline = next(
        result
        for result in results
        if result["configuration"] == "kmeans"
    )

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
                "sklearn_map": result["sklearn_map"],
                "delta_sklearn_map": (
                    result["sklearn_map"]
                    - baseline["sklearn_map"]
                ),
                "voc_2007_11_point_map": (
                    result["voc_2007_11_point_map"]
                ),
                "delta_voc07_map": (
                    result["voc_2007_11_point_map"]
                    - baseline["voc_2007_11_point_map"]
                ),
                "label_accuracy": (
                    result["label_accuracy_at_threshold"]
                ),
                "delta_label_accuracy": (
                    result["label_accuracy_at_threshold"]
                    - baseline[
                        "label_accuracy_at_threshold"
                    ]
                ),
                "micro_f1": result["micro_f1"],
                "delta_micro_f1": (
                    result["micro_f1"]
                    - baseline["micro_f1"]
                ),
                "macro_f1": result["macro_f1"],
                "delta_macro_f1": (
                    result["macro_f1"]
                    - baseline["macro_f1"]
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
        "checkpoint": str(args.checkpoint),
        "n_shot": args.n_shot,
        "baseline": {
            key: baseline[key]
            for key in [
                "mean_support_diversity",
                "sklearn_map",
                "voc_2007_11_point_map",
                "label_accuracy_at_threshold",
                "micro_f1",
                "macro_f1",
            ]
        },
        "random_mean": {
            "mean_support_diversity": float(
                np.mean(
                    [
                        result["mean_support_diversity"]
                        for result in random_results
                    ]
                )
            ),
            "sklearn_map": float(
                np.mean(
                    [
                        result["sklearn_map"]
                        for result in random_results
                    ]
                )
            ),
            "sklearn_map_std": float(
                np.std(
                    [
                        result["sklearn_map"]
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

    print("\nSupport-selection experiment complete")
    print(f"Comparison: {comparison_path}")
    print(
        f"Summary: {args.output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
