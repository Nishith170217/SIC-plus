import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Oxford Pets SIC results."
    )
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def save_training_curves(history, output_dir):
    epochs = [entry["epoch"] for entry in history]
    train_loss = [
        entry["train_loss"] for entry in history
    ]
    validation_loss = [
        entry["loss"] for entry in history
    ]
    top1 = [
        entry["top1_accuracy"] for entry in history
    ]
    top5 = [
        entry["top5_accuracy"] for entry in history
    ]
    macro_f1 = [
        entry["macro_f1"] for entry in history
    ]
    balanced = [
        entry["balanced_accuracy"] for entry in history
    ]

    best_index = int(np.argmax(top1))
    best_epoch = epochs[best_index]
    best_top1 = top1[best_index]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    axes[0].plot(
        epochs,
        train_loss,
        label="Training loss",
        linewidth=2,
    )
    axes[0].plot(
        epochs,
        validation_loss,
        label="Validation loss",
        linewidth=2,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Oxford Pets SIC loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        top1,
        label="Top-1 accuracy",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        top5,
        label="Top-5 accuracy",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        macro_f1,
        label="Macro F1",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        balanced,
        label="Balanced accuracy",
        linewidth=2,
    )
    axes[1].scatter(
        [best_epoch],
        [best_top1],
        color="black",
        zorder=5,
        label=(
            f"Best Top-1: {best_top1:.2f}% "
            f"(epoch {best_epoch})"
        ),
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Oxford Pets SIC validation metrics")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.savefig(
        output_dir / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "training_curves.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return best_epoch, best_top1


def save_per_class_analysis(evaluation, output_dir):
    rows = []

    for class_name, result in evaluation[
        "per_class"
    ].items():
        rows.append({
            "class_name": class_name,
            "class_index": result["class_index"],
            "test_images": result["test_images"],
            "accuracy": result["accuracy"],
        })

    rows.sort(key=lambda row: row["accuracy"])

    csv_path = output_dir / "per_class_accuracy.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "class_index",
                "test_images",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    class_names = [
        row["class_name"].replace("_", " ")
        for row in rows
    ]
    accuracies = [
        row["accuracy"] for row in rows
    ]

    colors = [
        "tab:red"
        if accuracy < 85
        else "tab:orange"
        if accuracy < 90
        else "tab:green"
        for accuracy in accuracies
    ]

    figure, axis = plt.subplots(
        figsize=(11, 12),
        constrained_layout=True,
    )

    positions = np.arange(len(rows))

    axis.barh(
        positions,
        accuracies,
        color=colors,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlim(0, 105)
    axis.set_xlabel("Per-class accuracy (%)")
    axis.set_title(
        "Oxford-IIIT Pet SIC per-class accuracy"
    )
    axis.axvline(
        evaluation["balanced_accuracy"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Balanced accuracy: "
            f"{evaluation['balanced_accuracy']:.2f}%"
        ),
    )
    axis.grid(
        axis="x",
        alpha=0.25,
    )
    axis.legend()

    for position, accuracy in zip(
        positions,
        accuracies,
    ):
        axis.text(
            accuracy + 0.5,
            position,
            f"{accuracy:.1f}",
            va="center",
            fontsize=7,
        )

    figure.savefig(
        output_dir / "per_class_accuracy.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "per_class_accuracy.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.history.open() as handle:
        history = json.load(handle)

    with args.evaluation.open() as handle:
        evaluation = json.load(handle)

    best_epoch, best_top1 = save_training_curves(
        history,
        args.output_dir,
    )

    class_rows = save_per_class_analysis(
        evaluation,
        args.output_dir,
    )

    hardest = class_rows[:5]
    easiest = list(reversed(class_rows[-5:]))

    summary = {
        "dataset": evaluation["dataset"],
        "test_images": evaluation["test_images"],
        "n_classes": evaluation["n_classes"],
        "n_shot": evaluation["n_shot"],
        "best_epoch": best_epoch,
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
        "hardest_classes": hardest,
        "easiest_classes": easiest,
    }

    with (
        args.output_dir / "analysis_summary.json"
    ).open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("Oxford Pets analysis complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Top-1: {best_top1:.2f}%")

    print("\nFive hardest classes:")
    for row in hardest:
        print(
            f"  {row['class_name']:30s} "
            f"{row['accuracy']:.2f}%"
        )

    print("\nFive easiest classes:")
    for row in easiest:
        print(
            f"  {row['class_name']:30s} "
            f"{row['accuracy']:.2f}%"
        )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
