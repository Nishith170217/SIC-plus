import sys
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from sic import SIC
from voc_dataset import VOC_CLASSES, get_voc_dataloader


PAIR_NAMES = ["S1-S2", "S1-S3", "S2-S3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse SIC prototype diversity on Pascal VOC 2007."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--support_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def build_model(support_loader, checkpoint, device):
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

    return model


def pairwise_metrics(features):
    cosine_values = []
    euclidean_values = []

    for first, second in combinations(range(len(features)), 2):
        cosine = functional.cosine_similarity(
            features[first].unsqueeze(0),
            features[second].unsqueeze(0),
        ).item()

        euclidean = torch.linalg.vector_norm(
            features[first] - features[second]
        ).item()

        cosine_values.append(cosine)
        euclidean_values.append(euclidean)

    return cosine_values, euclidean_values


def positive_classes(target):
    return [
        VOC_CLASSES[index]
        for index in torch.where(target > 0)[0].tolist()
    ]


def pearson_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")

    return float(np.corrcoef(first, second)[0, 1])


def save_csv(rows, output_path):
    fieldnames = [
        "class_index",
        "class_name",
        "prototype_1_index",
        "prototype_2_index",
        "prototype_3_index",
        "prototype_1_image_id",
        "prototype_2_image_id",
        "prototype_3_image_id",
        "prototype_1_labels",
        "prototype_2_labels",
        "prototype_3_labels",
        "unique_prototype_count",
        "cosine_s1_s2",
        "cosine_s1_s3",
        "cosine_s2_s3",
        "mean_cosine_similarity",
        "min_cosine_similarity",
        "max_cosine_similarity",
        "diversity",
        "euclidean_s1_s2",
        "euclidean_s1_s3",
        "euclidean_s2_s3",
        "mean_euclidean_distance",
        "min_euclidean_distance",
        "max_euclidean_distance",
        "sklearn_ap",
        "voc_2007_11_point_ap",
        "positive_count",
    ]

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_similarity_heatmap(rows, output_path):
    matrix = np.asarray(
        [
            [
                row["cosine_s1_s2"],
                row["cosine_s1_s3"],
                row["cosine_s2_s3"],
            ]
            for row in rows
        ]
    )

    figure, axis = plt.subplots(figsize=(7, 10))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    axis.set_xticks(range(3))
    axis.set_xticklabels(PAIR_NAMES)
    axis.set_yticks(range(len(VOC_CLASSES)))
    axis.set_yticklabels(VOC_CLASSES)
    axis.set_title("VOC support-prototype cosine similarity")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Cosine similarity",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_diversity_scatter(rows, output_path):
    diversity = np.asarray(
        [row["diversity"] for row in rows]
    )
    average_precision = np.asarray(
        [row["sklearn_ap"] for row in rows]
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(
        diversity,
        average_precision,
        s=55,
        color="tab:blue",
    )

    for row in rows:
        axis.annotate(
            row["class_name"],
            (row["diversity"], row["sklearn_ap"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    correlation = pearson_correlation(
        diversity,
        average_precision,
    )

    axis.set_xlabel(
        "Prototype diversity (1 - mean cosine similarity)"
    )
    axis.set_ylabel("Per-class sklearn AP (%)")
    axis.set_title(
        "Support-prototype diversity versus Pascal VOC AP\n"
        f"Pearson correlation: {correlation:.3f}"
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Loading Pascal VOC support set...")

    _, _, support_loader = get_voc_dataloader(
        args.data_dir,
        batch_size=8,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=8,
    )

    support_dataset = support_loader.dataset

    print("Loading trained SIC model...")
    model = build_model(
        support_loader,
        args.checkpoint,
        device,
    )

    print("Computing support features and prototypes...")
    with torch.no_grad():
        model.precompute()

    cluster_features = model.cluster_feat.detach().cpu()
    cluster_labels = model.cluster_y.detach().cpu()
    cluster_indices = model.cluster_indices.detach().cpu()

    expected_prototypes = len(VOC_CLASSES) * model.n_shot

    if len(cluster_features) != expected_prototypes:
        raise RuntimeError(
            f"Expected {expected_prototypes} prototypes, "
            f"found {len(cluster_features)}"
        )

    with open(args.evaluation) as file:
        evaluation = json.load(file)

    rows = []

    for class_index, class_name in enumerate(VOC_CLASSES):
        start = class_index * model.n_shot
        end = start + model.n_shot

        features = cluster_features[start:end]
        labels = cluster_labels[start:end]
        indices = cluster_indices[start:end].tolist()

        if not torch.all(labels == class_index):
            raise RuntimeError(
                f"Prototype-label mismatch for {class_name}"
            )

        cosine_values, euclidean_values = pairwise_metrics(
            features
        )

        image_ids = [
            support_dataset.image_ids[index]
            for index in indices
        ]

        support_labels = [
            positive_classes(support_dataset.labels[index])
            for index in indices
        ]

        class_evaluation = evaluation["per_class"][class_name]
        mean_cosine = float(np.mean(cosine_values))

        row = {
            "class_index": class_index,
            "class_name": class_name,
            "prototype_1_index": indices[0],
            "prototype_2_index": indices[1],
            "prototype_3_index": indices[2],
            "prototype_1_image_id": image_ids[0],
            "prototype_2_image_id": image_ids[1],
            "prototype_3_image_id": image_ids[2],
            "prototype_1_labels": json.dumps(support_labels[0]),
            "prototype_2_labels": json.dumps(support_labels[1]),
            "prototype_3_labels": json.dumps(support_labels[2]),
            "unique_prototype_count": len(set(indices)),
            "cosine_s1_s2": cosine_values[0],
            "cosine_s1_s3": cosine_values[1],
            "cosine_s2_s3": cosine_values[2],
            "mean_cosine_similarity": mean_cosine,
            "min_cosine_similarity": float(
                np.min(cosine_values)
            ),
            "max_cosine_similarity": float(
                np.max(cosine_values)
            ),
            "diversity": 1.0 - mean_cosine,
            "euclidean_s1_s2": euclidean_values[0],
            "euclidean_s1_s3": euclidean_values[1],
            "euclidean_s2_s3": euclidean_values[2],
            "mean_euclidean_distance": float(
                np.mean(euclidean_values)
            ),
            "min_euclidean_distance": float(
                np.min(euclidean_values)
            ),
            "max_euclidean_distance": float(
                np.max(euclidean_values)
            ),
            "sklearn_ap": class_evaluation["sklearn_ap"],
            "voc_2007_11_point_ap": class_evaluation[
                "voc_2007_11_point_ap"
            ],
            "positive_count": class_evaluation[
                "positive_count"
            ],
        }
        rows.append(row)

    diversity_values = [
        row["diversity"] for row in rows
    ]
    sklearn_ap_values = [
        row["sklearn_ap"] for row in rows
    ]
    voc_ap_values = [
        row["voc_2007_11_point_ap"] for row in rows
    ]

    ranked_rows = sorted(
        rows,
        key=lambda row: row["diversity"],
        reverse=True,
    )

    summary = {
        "dataset": "Pascal VOC 2007 trainval supports",
        "checkpoint": str(args.checkpoint),
        "n_classes": len(VOC_CLASSES),
        "n_shot": model.n_shot,
        "n_prototypes": len(cluster_features),
        "diversity_definition": (
            "1 - mean pairwise cosine similarity"
        ),
        "mean_diversity": float(
            np.mean(diversity_values)
        ),
        "minimum_diversity_class": ranked_rows[-1][
            "class_name"
        ],
        "minimum_diversity": ranked_rows[-1][
            "diversity"
        ],
        "maximum_diversity_class": ranked_rows[0][
            "class_name"
        ],
        "maximum_diversity": ranked_rows[0][
            "diversity"
        ],
        "pearson_diversity_vs_sklearn_ap": (
            pearson_correlation(
                diversity_values,
                sklearn_ap_values,
            )
        ),
        "pearson_diversity_vs_voc07_ap": (
            pearson_correlation(
                diversity_values,
                voc_ap_values,
            )
        ),
        "classes_ranked_by_diversity": [
            {
                "class_name": row["class_name"],
                "diversity": row["diversity"],
                "sklearn_ap": row["sklearn_ap"],
            }
            for row in ranked_rows
        ],
    }

    csv_path = args.output_dir / "support_metrics.csv"
    summary_path = args.output_dir / "summary.json"
    heatmap_path = (
        args.output_dir / "similarity_heatmap.png"
    )
    scatter_path = (
        args.output_dir / "diversity_vs_ap.png"
    )

    save_csv(rows, csv_path)

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    save_similarity_heatmap(rows, heatmap_path)
    save_diversity_scatter(rows, scatter_path)

    print("\nSupport-diversity analysis complete")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Scatter plot: {scatter_path}")
    print(
        "Diversity/AP Pearson correlation: "
        f"{summary['pearson_diversity_vs_sklearn_ap']:.3f}"
    )
    print(
        "Most diverse class: "
        f"{summary['maximum_diversity_class']}"
    )
    print(
        "Most redundant class: "
        f"{summary['minimum_diversity_class']}"
    )


if __name__ == "__main__":
    main()
