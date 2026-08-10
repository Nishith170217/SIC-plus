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
from tqdm import tqdm

from bcos import BcosEncoderWrapper, resnet50_long
from dogs_dataset import EvalTransform, get_dogs_dataloader
from sic import SIC


N_CLASSES = 120

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
            "Visualize SIC explanations under Stanford "
            "Dogs corruptions."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=-1)
    parser.add_argument("--min_confidence", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
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


def build_model(support_loader, checkpoint, device):
    featurizer = BcosEncoderWrapper(
        resnet50_long(pretrained=False)
    )

    model = SIC(
        featurizer=featurizer,
        n_classes=N_CLASSES,
        proj_dim=128,
        n_way=30,
        n_shot=3,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=False,
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


def find_sample(
    model,
    validation_loader,
    device,
    requested_index,
    min_confidence,
):
    if requested_index >= 0:
        return requested_index

    fallback_index = None
    offset = 0

    print("Finding a correctly classified Dogs sample...")

    with torch.inference_mode():
        for images, targets in tqdm(validation_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)

            logits = model.predict(images)
            predictions = logits.argmax(dim=1)
            probabilities = torch.sigmoid(logits)

            for local_index in range(len(targets)):
                if predictions[local_index] != targets[local_index]:
                    continue

                dataset_index = offset + local_index

                if fallback_index is None:
                    fallback_index = dataset_index

                predicted_class = int(
                    predictions[local_index]
                )
                confidence = float(
                    probabilities[
                        local_index,
                        predicted_class,
                    ]
                )

                if confidence >= min_confidence:
                    return dataset_index

            offset += len(targets)

    if fallback_index is not None:
        return fallback_index

    raise RuntimeError(
        "No correctly classified validation sample found"
    )


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
        predicted_class = int(logits.argmax(dim=1))
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

    return (
        predicted_class,
        probability,
        explanation["explanation"],
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    _, validation_loader, support_loader = (
        get_dogs_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_bcos=True,
            support_batch_size=args.support_batch_size,
            val_batch_size=args.batch_size,
        )
    )

    model = build_model(
        support_loader,
        args.checkpoint,
        device,
    )

    sample_index = find_sample(
        model,
        validation_loader,
        device,
        args.sample_index,
        args.min_confidence,
    )

    validation_dataset = validation_loader.dataset

    if not 0 <= sample_index < len(validation_dataset):
        raise IndexError(
            f"sample_index must be between 0 and "
            f"{len(validation_dataset) - 1}"
        )

    relative_path = Path(
        validation_dataset.file_paths[sample_index]
    )
    image_path = (
        validation_dataset.basepath / relative_path
    )
    target_class = int(
        validation_dataset.targets[sample_index]
    )
    class_name = relative_path.parent.name

    with Image.open(image_path) as image:
        original_image = image.convert("RGB")

    transform = EvalTransform(is_bcos=True)

    print(
        f"Selected sample={sample_index}, "
        f"class={class_name}, target={target_class}"
    )

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
                seed=args.seed + sample_index,
            )

            transformed = transform(corrupted_image)

            (
                predicted_class,
                probability,
                explanation,
            ) = predict_and_explain(
                model,
                transformed,
                target_class,
                args.percentile,
                args.smooth,
            )

            status = (
                "correct"
                if predicted_class == target_class
                else f"wrong: {predicted_class}"
            )

            axes[0, column].imshow(corrupted_image)
            axes[0, column].set_title(
                f"{label}\n"
                f"Target probability={probability:.3f}\n"
                f"Prediction: {status}"
            )
            axes[0, column].axis("off")

            axes[1, column].imshow(explanation)
            axes[1, column].set_title(
                "SIC contribution"
            )
            axes[1, column].axis("off")

        figure.suptitle(
            f"Dogs robustness explanation: {class_name}\n"
            f"sample={sample_index:04d}, "
            f"target={target_class}",
            fontsize=14,
        )
        figure.tight_layout(
            rect=[0, 0, 1, 0.91],
            h_pad=2.0,
        )

        output_path = (
            args.output_dir
            / (
                f"sample_{sample_index:04d}_"
                f"{class_name}_{group_name}.png"
            )
        )

        figure.savefig(output_path, dpi=200)
        plt.close(figure)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
