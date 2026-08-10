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

METRICS = [
    ("sklearn_map", "mAP", "tab:blue"),
    ("micro_f1", "Micro F1", "tab:orange"),
    ("macro_f1", "Macro F1", "tab:green"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.comparison) as file:
        rows = list(csv.DictReader(file))

    clean = next(
        row for row in rows
        if row["corruption"] == "clean"
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
        corruption_rows = [
            row for row in rows
            if row["corruption"] == corruption
        ]
        corruption_rows.sort(
            key=lambda row: float(row["severity"])
        )

        severities = [0.0] + [
            float(row["severity"])
            for row in corruption_rows
        ]

        for metric_key, metric_name, color in METRICS:
            values = [float(clean[metric_key])] + [
                float(row[metric_key])
                for row in corruption_rows
            ]

            axis.plot(
                severities,
                values,
                marker="o",
                linewidth=2,
                markersize=6,
                label=metric_name,
                color=color,
            )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3)
        axis.set_ylim(20, 90)

    axes[0].set_ylabel("Performance (%)")
    axes[-1].legend(
        loc="lower left",
        frameon=True,
    )

    figure.suptitle(
        "SIC robustness on Pascal VOC 2007",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
