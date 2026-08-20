import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from eval_pets import get_class_names, load_model
from pets_dataset import get_pets_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Oxford Pets confusion analysis."
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
        "--output_dir",
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
    return parser.parse_args()


def save_matrix_csv(path, matrix, class_names):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class"] + class_names)

        for class_name, row in zip(
            class_names,
            matrix,
        ):
            writer.writerow(
                [class_name] + row.tolist()
            )


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
    model = load_model(
        args,
        support_loader,
        device,
    )

    print("Precomputing support prototypes...")
    with torch.no_grad():
        model.precompute()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in tqdm(
            validation_loader,
            desc="Computing confusion matrix",
        ):
            logits = model.predict(images.to(device))
            predictions = logits.argmax(dim=1).cpu()

            all_targets.append(targets.cpu())
            all_predictions.append(predictions)

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()

    labels = np.arange(len(class_names))

    counts = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )

    row_totals = counts.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )

    save_matrix_csv(
        args.output_dir / "confusion_matrix_counts.csv",
        counts,
        class_names,
    )

    save_matrix_csv(
        args.output_dir
        / "confusion_matrix_normalized.csv",
        normalized,
        class_names,
    )

    confusion_rows = []

    for true_index in range(len(class_names)):
        for predicted_index in range(
            len(class_names)
        ):
            if true_index == predicted_index:
                continue

            count = int(
                counts[true_index, predicted_index]
            )

            if count == 0:
                continue

            rate = float(
                normalized[
                    true_index,
                    predicted_index,
                ]
                * 100.0
            )

            confusion_rows.append({
                "true_class": class_names[true_index],
                "predicted_class": (
                    class_names[predicted_index]
                ),
                "count": count,
                "true_class_error_rate": rate,
            })

    confusion_rows.sort(
        key=lambda row: (
            row["count"],
            row["true_class_error_rate"],
        ),
        reverse=True,
    )

    with (
        args.output_dir / "top_confusions.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "true_class",
                "predicted_class",
                "count",
                "true_class_error_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    display_names = [
        name.replace("_", " ")
        for name in class_names
    ]

    figure, axis = plt.subplots(
        figsize=(18, 16),
        constrained_layout=True,
    )

    image = axis.imshow(
        normalized * 100.0,
        cmap="Blues",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )

    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(
        display_names,
        rotation=90,
        fontsize=7,
    )
    axis.set_yticklabels(
        display_names,
        fontsize=7,
    )
    axis.set_xlabel("Predicted breed")
    axis.set_ylabel("True breed")
    axis.set_title(
        "Oxford-IIIT Pet SIC normalized confusion matrix"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Percentage of true-class images (%)"
    )

    figure.savefig(
        args.output_dir
        / "confusion_matrix_normalized.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir
        / "confusion_matrix_normalized.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "test_images": int(len(targets)),
        "correct_predictions": int(
            (targets == predictions).sum()
        ),
        "incorrect_predictions": int(
            (targets != predictions).sum()
        ),
        "top_confusions": confusion_rows[:20],
    }

    with (
        args.output_dir
        / "confusion_summary.json"
    ).open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nMost frequent confusions:")
    for row in confusion_rows[:15]:
        print(
            f"  {row['true_class']:28s} -> "
            f"{row['predicted_class']:28s} "
            f"count={row['count']:2d}, "
            f"rate={row['true_class_error_rate']:.2f}%"
        )

    print(
        f"\nIncorrect predictions: "
        f"{summary['incorrect_predictions']}/"
        f"{summary['test_images']}"
    )
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
