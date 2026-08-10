import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CORRUPTIONS = [
    ("gaussian_noise", "Gaussian noise", "Noise σ"),
    ("gaussian_blur", "Gaussian blur", "Blur radius"),
    (
        "central_occlusion",
        "Central occlusion",
        "Occluded area fraction",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_csv", type=Path, required=True)
    parser.add_argument("--dogs_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def retained_performance(value, clean):
    return 100.0 * value / clean


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    voc_rows = read_csv(args.voc_csv)
    dogs_rows = read_csv(args.dogs_csv)

    voc_clean = float(
        next(
            row for row in voc_rows
            if row["corruption"] == "clean"
        )["sklearn_map"]
    )
    dogs_clean = float(
        next(
            row for row in dogs_rows
            if row["corruption"] == "clean"
        )["top1_accuracy"]
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        sharey=True,
    )

    for axis, (
        corruption,
        title,
        xlabel,
    ) in zip(axes, CORRUPTIONS):
        voc_corrupted = sorted(
            [
                row for row in voc_rows
                if row["corruption"] == corruption
            ],
            key=lambda row: float(row["severity"]),
        )
        dogs_corrupted = sorted(
            [
                row for row in dogs_rows
                if row["corruption"] == corruption
            ],
            key=lambda row: float(row["severity"]),
        )

        severities = [0.0] + [
            float(row["severity"])
            for row in voc_corrupted
        ]

        voc_retained = [100.0] + [
            retained_performance(
                float(row["sklearn_map"]),
                voc_clean,
            )
            for row in voc_corrupted
        ]

        dogs_retained = [100.0] + [
            retained_performance(
                float(row["top1_accuracy"]),
                dogs_clean,
            )
            for row in dogs_corrupted
        ]

        axis.plot(
            severities,
            voc_retained,
            marker="o",
            linewidth=2,
            label="VOC mAP",
            color="tab:blue",
        )
        axis.plot(
            severities,
            dogs_retained,
            marker="s",
            linewidth=2,
            label="Dogs Top-1",
            color="tab:red",
        )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3)
        axis.set_ylim(20, 105)

    axes[0].set_ylabel(
        "Clean performance retained (%)"
    )
    axes[-1].legend(loc="lower left")

    figure.suptitle(
        "SIC corruption robustness across datasets",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
