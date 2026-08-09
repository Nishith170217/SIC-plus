import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from voc_dataset import VOC_CLASSES, VOCDataset


DEFAULT_CLASSES = [
    "sheep",
    "person",
    "bottle",
    "train",
    "chair",
    "horse",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot VOC support-selection comparisons."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
    )
    parser.add_argument("--n_shot", type=int, default=3)
    return parser.parse_args()


def load_result(path):
    with open(path) as file:
        return json.load(file)


def positive_labels(dataset, index):
    target = dataset.labels[index]

    return [
        VOC_CLASSES[class_index]
        for class_index in range(len(VOC_CLASSES))
        if target[class_index] > 0
    ]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VOCDataset(
        args.data_dir,
        split="trainval",
        transform=None,
    )

    strategies = [
        (
            "K-means",
            load_result(args.results_dir / "kmeans.json"),
        ),
        (
            "Compact representative",
            load_result(
                args.results_dir / "max_similarity.json"
            ),
        ),
        (
            "Maximum diversity",
            load_result(
                args.results_dir / "max_diversity.json"
            ),
        ),
    ]

    for class_name in args.classes:
        if class_name not in VOC_CLASSES:
            raise ValueError(
                f"Unknown VOC class: {class_name}"
            )

        class_index = VOC_CLASSES.index(class_name)
        start = class_index * args.n_shot
        end = start + args.n_shot

        figure, axes = plt.subplots(
            len(strategies),
            args.n_shot,
            figsize=(12, 10),
        )

        baseline_ap = strategies[0][1]["per_class"][
            class_name
        ]["sklearn_ap"]

        for row, (strategy_name, result) in enumerate(
            strategies
        ):
            indices = result["selected_indices"][start:end]
            strategy_ap = result["per_class"][class_name][
                "sklearn_ap"
            ]
            delta_ap = strategy_ap - baseline_ap

            for column, dataset_index in enumerate(indices):
                image_id = dataset.image_ids[dataset_index]
                image_path = (
                    dataset.image_dir / f"{image_id}.jpg"
                )

                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    axes[row, column].imshow(image)

                labels = positive_labels(
                    dataset,
                    dataset_index,
                )

                axes[row, column].set_title(
                    f"S{column + 1}: {image_id}\n"
                    f"Labels: {', '.join(labels)}",
                    fontsize=8,
                )
                axes[row, column].axis("off")

                if column == 0:
                    axes[row, column].set_ylabel(
                        f"{strategy_name}\n"
                        f"AP={strategy_ap:.2f}%\n"
                        f"Δ={delta_ap:+.2f}",
                        fontsize=10,
                    )

        figure.suptitle(
            f"VOC support comparison: {class_name}",
            fontsize=15,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = (
            args.output_dir
            / f"{class_name}_support_comparison.png"
        )
        figure.savefig(output_path, dpi=200)
        plt.close(figure)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
