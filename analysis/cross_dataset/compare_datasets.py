import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DATASET_METADATA = {
    "Stanford Dogs": {
        "task": "single-label",
        "classes": 120,
        "train_images": 12000,
        "test_images": 8580,
        "backbone": "B-cos ResNet50",
        "n_shot": 3,
    },
    "Oxford-IIIT Pet": {
        "task": "single-label",
        "classes": 37,
        "train_images": 3680,
        "test_images": 3669,
        "backbone": "B-cos ResNet50",
        "n_shot": 3,
    },
    "Pascal VOC 2007": {
        "task": "multi-label",
        "classes": 20,
        "train_images": 5011,
        "test_images": 4952,
        "backbone": "B-cos DenseNet121",
        "n_shot": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SIC performance across datasets."
    )
    parser.add_argument(
        "--dogs",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--voc",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pets",
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


def add_bar_labels(axis, bars):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_performance_figure(
    dogs_metrics,
    pets_metrics,
    voc_metrics,
    output_dir,
):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        constrained_layout=True,
    )

    single_metrics = [
        "Top-1",
        "Top-5",
        "Macro F1",
        "Balanced accuracy",
    ]

    dogs_values = [
        dogs_metrics["top1_accuracy"],
        dogs_metrics["top5_accuracy"],
        dogs_metrics["macro_f1"],
        dogs_metrics["balanced_accuracy"],
    ]

    pets_values = [
        pets_metrics["top1_accuracy"],
        pets_metrics["top5_accuracy"],
        pets_metrics["macro_f1"],
        pets_metrics["balanced_accuracy"],
    ]

    positions = np.arange(len(single_metrics))
    width = 0.36

    dogs_bars = axes[0].bar(
        positions - width / 2,
        dogs_values,
        width,
        label="Stanford Dogs",
        color="tab:blue",
    )
    pets_bars = axes[0].bar(
        positions + width / 2,
        pets_values,
        width,
        label="Oxford-IIIT Pet",
        color="tab:orange",
    )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        single_metrics,
        rotation=15,
        ha="right",
    )
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title(
        "Single-label classification"
    )
    axes[0].grid(
        axis="y",
        alpha=0.25,
    )
    axes[0].legend()
    add_bar_labels(axes[0], dogs_bars)
    add_bar_labels(axes[0], pets_bars)

    voc_metric_names = [
        "Sklearn\nmAP",
        "VOC07\nmAP",
        "Micro F1",
        "Macro F1",
        "Label accuracy",
    ]

    voc_values = [
        voc_metrics["sklearn_map"],
        voc_metrics["voc_2007_11_point_map"],
        voc_metrics["micro_f1"],
        voc_metrics["macro_f1"],
        voc_metrics["label_accuracy_at_threshold"],
    ]

    voc_colors = [
        "tab:green",
        "tab:green",
        "tab:purple",
        "tab:purple",
        "tab:gray",
    ]

    voc_bars = axes[1].bar(
        np.arange(len(voc_metric_names)),
        voc_values,
        color=voc_colors,
    )

    axes[1].set_xticks(
        np.arange(len(voc_metric_names))
    )
    axes[1].set_xticklabels(voc_metric_names)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title(
        "Pascal VOC 2007 multi-label classification"
    )
    axes[1].grid(
        axis="y",
        alpha=0.25,
    )
    add_bar_labels(axes[1], voc_bars)

    figure.suptitle(
        "SIC cross-dataset performance comparison",
        fontsize=16,
    )

    figure.savefig(
        output_dir / "cross_dataset_performance.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "cross_dataset_performance.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_dataset_scale_figure(output_dir):
    names = list(DATASET_METADATA.keys())
    classes = [
        DATASET_METADATA[name]["classes"]
        for name in names
    ]
    train_images = [
        DATASET_METADATA[name]["train_images"]
        for name in names
    ]
    test_images = [
        DATASET_METADATA[name]["test_images"]
        for name in names
    ]

    positions = np.arange(len(names))
    width = 0.36

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )

    class_bars = axes[0].bar(
        positions,
        classes,
        color=[
            "tab:blue",
            "tab:orange",
            "tab:green",
        ],
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        names,
        rotation=15,
        ha="right",
    )
    axes[0].set_ylabel("Number of classes")
    axes[0].set_title("Dataset class complexity")
    axes[0].grid(axis="y", alpha=0.25)

    for bar, value in zip(class_bars, classes):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            str(value),
            ha="center",
        )

    train_bars = axes[1].bar(
        positions - width / 2,
        train_images,
        width,
        label="Training",
        color="tab:cyan",
    )
    test_bars = axes[1].bar(
        positions + width / 2,
        test_images,
        width,
        label="Test",
        color="tab:pink",
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        names,
        rotation=15,
        ha="right",
    )
    axes[1].set_ylabel("Number of images")
    axes[1].set_title("Dataset split sizes")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    for bars in [train_bars, test_bars]:
        for bar in bars:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 150,
                f"{int(bar.get_height()):,}",
                ha="center",
                fontsize=8,
            )

    figure.savefig(
        output_dir / "dataset_characteristics.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "dataset_characteristics.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dogs = load_json(args.dogs)
    voc = load_json(args.voc)
    pets = load_json(args.pets)

    dogs_metrics = dogs["baseline"]

    rows = [
        {
            **DATASET_METADATA["Stanford Dogs"],
            "dataset": "Stanford Dogs",
            "primary_metric": "Top-1 accuracy",
            "primary_score": dogs_metrics[
                "top1_accuracy"
            ],
            "top1_accuracy": dogs_metrics[
                "top1_accuracy"
            ],
            "top5_accuracy": dogs_metrics[
                "top5_accuracy"
            ],
            "map": "",
            "voc07_map": "",
            "micro_f1": "",
            "macro_f1": dogs_metrics["macro_f1"],
            "balanced_accuracy": dogs_metrics[
                "balanced_accuracy"
            ],
            "label_accuracy": "",
        },
        {
            **DATASET_METADATA["Oxford-IIIT Pet"],
            "dataset": "Oxford-IIIT Pet",
            "primary_metric": "Top-1 accuracy",
            "primary_score": pets["top1_accuracy"],
            "top1_accuracy": pets["top1_accuracy"],
            "top5_accuracy": pets["top5_accuracy"],
            "map": "",
            "voc07_map": "",
            "micro_f1": "",
            "macro_f1": pets["macro_f1"],
            "balanced_accuracy": pets[
                "balanced_accuracy"
            ],
            "label_accuracy": "",
        },
        {
            **DATASET_METADATA["Pascal VOC 2007"],
            "dataset": "Pascal VOC 2007",
            "primary_metric": "Sklearn mAP",
            "primary_score": voc["sklearn_map"],
            "top1_accuracy": "",
            "top5_accuracy": "",
            "map": voc["sklearn_map"],
            "voc07_map": voc[
                "voc_2007_11_point_map"
            ],
            "micro_f1": voc["micro_f1"],
            "macro_f1": voc["macro_f1"],
            "balanced_accuracy": "",
            "label_accuracy": voc[
                "label_accuracy_at_threshold"
            ],
        },
    ]

    fieldnames = [
        "dataset",
        "task",
        "classes",
        "train_images",
        "test_images",
        "backbone",
        "n_shot",
        "primary_metric",
        "primary_score",
        "top1_accuracy",
        "top5_accuracy",
        "map",
        "voc07_map",
        "micro_f1",
        "macro_f1",
        "balanced_accuracy",
        "label_accuracy",
    ]

    with (
        args.output_dir / "comparison.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    single_label_differences = {
        "top1_accuracy": (
            pets["top1_accuracy"]
            - dogs_metrics["top1_accuracy"]
        ),
        "top5_accuracy": (
            pets["top5_accuracy"]
            - dogs_metrics["top5_accuracy"]
        ),
        "macro_f1": (
            pets["macro_f1"]
            - dogs_metrics["macro_f1"]
        ),
        "balanced_accuracy": (
            pets["balanced_accuracy"]
            - dogs_metrics["balanced_accuracy"]
        ),
    }

    summary = {
        "datasets": rows,
        "pets_minus_dogs": (
            single_label_differences
        ),
        "interpretation_constraints": [
            (
                "Dogs and Oxford Pets are directly "
                "comparable single-label tasks using "
                "the same B-cos ResNet50 backbone."
            ),
            (
                "Pascal VOC is a multi-label task and "
                "must be evaluated primarily with mAP "
                "rather than Top-1 accuracy."
            ),
            (
                "VOC label accuracy is not comparable "
                "to single-label Top-1 accuracy because "
                "most class labels are negative."
            ),
            (
                "VOC also uses a different backbone, "
                "B-cos DenseNet121."
            ),
        ],
    }

    with (
        args.output_dir / "summary.json"
    ).open("w") as handle:
        json.dump(summary, handle, indent=2)

    save_performance_figure(
        dogs_metrics,
        pets,
        voc,
        args.output_dir,
    )
    save_dataset_scale_figure(args.output_dir)

    print("Cross-dataset comparison complete")
    print("\nOxford Pets minus Stanford Dogs:")

    for metric, difference in (
        single_label_differences.items()
    ):
        print(
            f"  {metric:20s}: "
            f"{difference:+.2f} percentage points"
        )

    print(
        f"\nVOC primary result: "
        f"{voc['sklearn_map']:.2f}% mAP"
    )
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
