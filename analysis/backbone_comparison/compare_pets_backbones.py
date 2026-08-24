import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BACKBONES = {
    "ResNet50": {
        "parameters": 23_753_024,
        "training_seconds": 25_583,
        "checkpoint_bytes": 95_326_152,
    },
    "DenseNet121": {
        "parameters": 7_052_512,
        "training_seconds": 20_005,
        "checkpoint_bytes": 28_793_883,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resnet_evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--densenet_evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resnet_history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--densenet_history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def add_labels(axis, bars):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resnet_eval = load_json(
        args.resnet_evaluation
    )
    densenet_eval = load_json(
        args.densenet_evaluation
    )
    resnet_history = load_json(
        args.resnet_history
    )
    densenet_history = load_json(
        args.densenet_history
    )

    evaluations = {
        "ResNet50": resnet_eval,
        "DenseNet121": densenet_eval,
    }

    metric_keys = [
        "top1_accuracy",
        "top5_accuracy",
        "macro_f1",
        "balanced_accuracy",
    ]
    metric_labels = [
        "Top-1",
        "Top-5",
        "Macro F1",
        "Balanced accuracy",
    ]

    rows = []

    for name in ["ResNet50", "DenseNet121"]:
        evaluation = evaluations[name]
        metadata = BACKBONES[name]

        rows.append({
            "backbone": name,
            "top1_accuracy": evaluation[
                "top1_accuracy"
            ],
            "top5_accuracy": evaluation[
                "top5_accuracy"
            ],
            "macro_f1": evaluation["macro_f1"],
            "balanced_accuracy": evaluation[
                "balanced_accuracy"
            ],
            "parameters": metadata["parameters"],
            "training_seconds": metadata[
                "training_seconds"
            ],
            "checkpoint_bytes": metadata[
                "checkpoint_bytes"
            ],
        })

    with (
        args.output_dir / "comparison.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    differences = {}

    for metric in metric_keys:
        differences[metric] = (
            densenet_eval[metric]
            - resnet_eval[metric]
        )

    differences["parameter_reduction_percent"] = (
        1
        - BACKBONES["DenseNet121"]["parameters"]
        / BACKBONES["ResNet50"]["parameters"]
    ) * 100

    differences["training_time_reduction_percent"] = (
        1
        - BACKBONES["DenseNet121"][
            "training_seconds"
        ]
        / BACKBONES["ResNet50"]["training_seconds"]
    ) * 100

    differences["checkpoint_reduction_percent"] = (
        1
        - BACKBONES["DenseNet121"][
            "checkpoint_bytes"
        ]
        / BACKBONES["ResNet50"]["checkpoint_bytes"]
    ) * 100

    summary = {
        "dataset": "Oxford-IIIT Pet",
        "controlled_settings": {
            "epochs": 50,
            "batch_size": 8,
            "accumulation_steps": 4,
            "effective_batch_size": 32,
            "learning_rate": 0.001,
            "n_way": 30,
            "n_shot": 3,
            "training_gpu": "NVIDIA A100 40GB",
        },
        "results": rows,
        "densenet_minus_resnet": differences,
    }

    with (
        args.output_dir / "summary.json"
    ).open("w") as handle:
        json.dump(summary, handle, indent=2)

    positions = np.arange(len(metric_keys))
    width = 0.36

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    resnet_values = [
        resnet_eval[key] for key in metric_keys
    ]
    densenet_values = [
        densenet_eval[key] for key in metric_keys
    ]

    resnet_bars = axes[0].bar(
        positions - width / 2,
        resnet_values,
        width,
        label="ResNet50",
        color="tab:blue",
    )
    densenet_bars = axes[0].bar(
        positions + width / 2,
        densenet_values,
        width,
        label="DenseNet121",
        color="tab:orange",
    )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        metric_labels,
        rotation=15,
        ha="right",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Predictive performance")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    add_labels(axes[0], resnet_bars)
    add_labels(axes[0], densenet_bars)

    efficiency_names = [
        "Parameters",
        "Training time",
        "Checkpoint size",
    ]
    resnet_efficiency = [100, 100, 100]
    densenet_efficiency = [
        (
            BACKBONES["DenseNet121"]["parameters"]
            / BACKBONES["ResNet50"]["parameters"]
            * 100
        ),
        (
            BACKBONES["DenseNet121"][
                "training_seconds"
            ]
            / BACKBONES["ResNet50"][
                "training_seconds"
            ]
            * 100
        ),
        (
            BACKBONES["DenseNet121"][
                "checkpoint_bytes"
            ]
            / BACKBONES["ResNet50"][
                "checkpoint_bytes"
            ]
            * 100
        ),
    ]

    efficiency_positions = np.arange(
        len(efficiency_names)
    )

    axes[1].bar(
        efficiency_positions - width / 2,
        resnet_efficiency,
        width,
        label="ResNet50 reference",
        color="tab:blue",
    )
    dense_efficiency_bars = axes[1].bar(
        efficiency_positions + width / 2,
        densenet_efficiency,
        width,
        label="DenseNet121",
        color="tab:orange",
    )

    axes[1].set_xticks(efficiency_positions)
    axes[1].set_xticklabels(efficiency_names)
    axes[1].set_ylabel(
        "Relative to ResNet50 (%)"
    )
    axes[1].set_title("Computational efficiency")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    for bar in dense_efficiency_bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{bar.get_height():.1f}%",
            ha="center",
            fontsize=8,
        )

    figure.suptitle(
        "SIC backbone comparison on Oxford-IIIT Pet",
        fontsize=15,
    )

    figure.savefig(
        args.output_dir / "backbone_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "backbone_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )

    axes[0].plot(
        [entry["epoch"] for entry in resnet_history],
        [
            entry["top1_accuracy"]
            for entry in resnet_history
        ],
        label="ResNet50",
        linewidth=2,
    )
    axes[0].plot(
        [
            entry["epoch"]
            for entry in densenet_history
        ],
        [
            entry["top1_accuracy"]
            for entry in densenet_history
        ],
        label="DenseNet121",
        linewidth=2,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Validation accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        [entry["epoch"] for entry in resnet_history],
        [entry["loss"] for entry in resnet_history],
        label="ResNet50",
        linewidth=2,
    )
    axes[1].plot(
        [
            entry["epoch"]
            for entry in densenet_history
        ],
        [
            entry["loss"]
            for entry in densenet_history
        ],
        label="DenseNet121",
        linewidth=2,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Validation loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.savefig(
        args.output_dir / "training_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "training_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("Backbone comparison complete")
    print(json.dumps(differences, indent=2))
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
