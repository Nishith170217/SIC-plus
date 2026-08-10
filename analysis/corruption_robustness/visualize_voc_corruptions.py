import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from dogs_dataset import EvalTransform
from sic import SIC
from voc_dataset import (
    VOC_CLASSES,
    VOCDataset,
    get_voc_dataloader,
)


CORRUPTION_GROUPS = {
    "gaussian_noise": [
        ("Clean", "clean", 0.0),
        ("Mild noise", "gaussian_noise", 0.05),
        ("Severe noise", "gaussian_noise", 0.20),
    ],
    "gaussian_blur": [
        ("Clean", "clean", 0.0),
        ("Mild blur", "gaussian_blur", 1.0),
        ("Severe blur", "gaussian_blur", 4.0),
    ],
    "central_occlusion": [
        ("Clean", "clean", 0.0),
        ("Mild occlusion", "central_occlusion", 0.10),
        ("Severe occlusion", "central_occlusion", 0.40),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize SIC explanations under VOC corruptions."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=6)
    parser.add_argument("--class_name", default="horse")
    parser.add_argument("--support_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--smooth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if args.class_name not in VOC_CLASSES:
        parser.error(
            f"Unknown VOC class: {args.class_name}"
        )

    return args


def apply_corruption(
    image,
    corruption,
    severity,
    seed,
):
    if corruption == "clean":
        return image.copy()

    if corruption == "gaussian_noise":
        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        generator = np.random.default_rng(seed)
        noise = generator.normal(
            0.0,
            severity,
            size=array.shape,
        ).astype(np.float32)

        array = np.clip(array + noise, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)

        return Image.fromarray(array, mode="RGB")

    if corruption == "gaussian_blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius=severity)
        )

    if corruption == "central_occlusion":
        corrupted = image.copy()
        width, height = corrupted.size

        side_fraction = np.sqrt(severity)
        occlusion_width = max(
            1,
            int(width * side_fraction),
        )
        occlusion_height = max(
            1,
            int(height * side_fraction),
        )

        left = (width - occlusion_width) // 2
        top = (height - occlusion_height) // 2
        right = left + occlusion_width
        bottom = top + occlusion_height

        drawer = ImageDraw.Draw(corrupted)
        drawer.rectangle(
            [left, top, right, bottom],
            fill=(127, 127, 127),
        )

        return corrupted

    raise ValueError(f"Unknown corruption: {corruption}")


def build_model(
    support_loader,
    checkpoint,
    device,
):
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

    print("Precomputing clean K-means supports...")

    with torch.no_grad():
        model.precompute()

    return model


def predict_and_explain(
    model,
    transformed_image,
    class_index,
    percentile,
    smooth,
):
    query = transformed_image.unsqueeze(0).to(
        model.device
    )

    with torch.no_grad():
        logits = model.predict(query)
        probability = float(
            torch.sigmoid(logits[0, class_index])
        )

    model.zero_grad(set_to_none=True)

    explanation = model.explain_prediction(
        query,
        class_index,
        alpha_percentile=percentile,
        smooth=smooth,
    )

    return probability, explanation["explanation"]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    raw_dataset = VOCDataset(
        args.data_dir,
        split="test",
        transform=None,
    )

    if not 0 <= args.sample_index < len(raw_dataset):
        raise IndexError(
            f"sample_index must be between 0 and "
            f"{len(raw_dataset) - 1}"
        )

    image_id = raw_dataset.image_ids[args.sample_index]
    image_path = (
        raw_dataset.image_dir / f"{image_id}.jpg"
    )

    with Image.open(image_path) as image:
        original_image = image.convert("RGB")

    class_index = VOC_CLASSES.index(args.class_name)
    target = int(
        raw_dataset.labels[
            args.sample_index,
            class_index,
        ].item()
    )

    _, _, support_loader = get_voc_dataloader(
        args.data_dir,
        batch_size=8,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=8,
    )

    model = build_model(
        support_loader,
        args.checkpoint,
        device,
    )

    transform = EvalTransform(is_bcos=True)

    for group_name, configurations in (
        CORRUPTION_GROUPS.items()
    ):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(12, 8),
        )

        for column, (
            label,
            corruption,
            severity,
        ) in enumerate(configurations):
            corrupted_image = apply_corruption(
                original_image,
                corruption,
                severity,
                seed=args.seed + args.sample_index,
            )

            transformed = transform(corrupted_image)

            probability, explanation = (
                predict_and_explain(
                    model,
                    transformed,
                    class_index,
                    args.percentile,
                    args.smooth,
                )
            )

            axes[0, column].imshow(corrupted_image)
            axes[0, column].set_title(
                f"{label}\n"
                f"Probability={probability:.3f}"
            )
            axes[0, column].axis("off")

            axes[1, column].imshow(explanation)
            axes[1, column].set_title(
                "SIC contribution"
            )
            axes[1, column].axis("off")

        figure.suptitle(
            f"VOC robustness explanation: "
            f"{args.class_name}\n"
            f"sample={args.sample_index:04d}, "
            f"image_id={image_id}, target={target}",
            fontsize=14,
        )
        figure.tight_layout(
            rect=[0, 0, 1, 0.92],
            h_pad=2.0,
        )

        output_path = (
            args.output_dir
            / (
                f"sample_{args.sample_index:04d}_"
                f"{args.class_name}_{group_name}.png"
            )
        )

        figure.savefig(output_path, dpi=200)
        plt.close(figure)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
