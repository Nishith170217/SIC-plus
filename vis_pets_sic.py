# SIC qualitative visualizations for Oxford-IIIT Pet.

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from bcos import BcosEncoderWrapper, resnet50_long
from pets_dataset import N_PET_CLASSES, get_pets_dataloader
from sic import SIC


def get_class_names(data_dir):
    class_names = [None] * N_PET_CLASSES
    split_path = Path(data_dir) / "annotations" / "trainval.txt"

    with split_path.open("r") as handle:
        for line in handle:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            image_name = parts[0]
            class_index = int(parts[1]) - 1
            class_names[class_index] = image_name.rsplit("_", 1)[0]

    return [
        name if name is not None else f"class_{index}"
        for index, name in enumerate(class_names)
    ]


def safe_name(value):
    return "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in value
    )


def plot_prediction(
    model,
    image,
    target,
    prediction,
    probability,
    class_names,
    percentile,
    smooth,
):
    n_columns = model.n_shot + 2

    figure, axes = plt.subplots(
        2,
        n_columns,
        figsize=(4 * n_columns, 8),
        constrained_layout=True,
    )

    target_name = class_names[target]
    prediction_name = class_names[prediction]
    correct = target == prediction
    status = "correct" if correct else "misclassified"
    color = "tab:blue" if correct else "tab:red"

    display_image = image[:3].permute(1, 2, 0).numpy()

    axes[0, 0].imshow(display_image)
    axes[0, 0].set_title(
        f"Test image\n"
        f"Target: {target_name}\n"
        f"Prediction: {prediction_name}"
    )
    axes[0, 0].axis("off")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.95,
        f"Status: {status}\n"
        f"Predicted probability: {probability:.3f}\n"
        f"Sample target index: {target}\n"
        f"Predicted class index: {prediction}",
        va="top",
        fontsize=11,
        color=color,
    )

    calibration_image = (
        image.unsqueeze(0)
        .clone()
        .to(model.device)
    )

    model.plot_calibration_scores(
        calibration_image,
        prediction,
        axes[0, 1],
        color=color,
        y_probs=False,
    )
    axes[0, 1].set_title(
        f"Support evidence\n{prediction_name}"
    )

    explanation_image = (
        image.unsqueeze(0)
        .clone()
        .to(model.device)
    )

    test_explanation = model.explain_prediction(
        explanation_image,
        prediction,
        alpha_percentile=percentile,
        smooth=smooth,
    )

    axes[1, 1].imshow(
        test_explanation["explanation"]
    )
    axes[1, 1].set_title("Test contribution")
    axes[1, 1].axis("off")

    for support_number in range(model.n_shot):
        prototype_index = (
            prediction * model.n_shot
            + support_number
        )

        prototype = model.explain_prototype(
            prototype_index,
            None,
            alpha_percentile=percentile,
            smooth=smooth,
        )

        column = support_number + 2

        axes[0, column].imshow(
            prototype["image"][:3].transpose(1, 2, 0)
        )
        axes[0, column].set_title(
            f"Support {support_number + 1}\n"
            f"{prediction_name}"
        )
        axes[0, column].axis("off")

        axes[1, column].imshow(
            prototype["explanation"]
        )
        axes[1, column].set_title(
            f"Support contribution "
            f"{support_number + 1}"
        )
        axes[1, column].axis("off")

    figure.suptitle(
        f"Oxford-IIIT Pet SIC explanation: "
        f"{prediction_name}",
        fontsize=18,
    )

    return figure, status, target_name, prediction_name


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate SIC explanations for Oxford-IIIT Pet."
        )
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
        "--sample_indices",
        type=int,
        nargs="+",
        default=[0],
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
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    if args.n_shot < 1:
        parser.error("--n_shot must be at least 1")

    if args.smooth < 1:
        parser.error("--smooth must be at least 1")

    return args


def main():
    args = parse_args()

    torch.multiprocessing.set_sharing_strategy(
        "file_system"
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

    featurizer = BcosEncoderWrapper(
        resnet50_long(pretrained=False)
    )

    model = SIC(
        featurizer=featurizer,
        n_classes=N_PET_CLASSES,
        proj_dim=128,
        n_way=30,
        n_shot=args.n_shot,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=False,
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Precomputing support prototypes...")
    with torch.no_grad():
        model.precompute()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = validation_loader.dataset

    for sample_index in args.sample_indices:
        if sample_index < 0 or sample_index >= len(dataset):
            print(
                f"Skipping invalid sample index "
                f"{sample_index}"
            )
            continue

        image, target = dataset[sample_index]
        target = int(target)

        with torch.no_grad():
            logits = model.predict(
                image.unsqueeze(0).to(device)
            )
            probabilities = torch.sigmoid(logits)
            prediction = int(
                logits.argmax(dim=1).item()
            )
            probability = float(
                probabilities[0, prediction].item()
            )

        figure, status, target_name, prediction_name = (
            plot_prediction(
                model=model,
                image=image,
                target=target,
                prediction=prediction,
                probability=probability,
                class_names=class_names,
                percentile=args.percentile,
                smooth=args.smooth,
            )
        )

        filename = (
            f"sample_{sample_index:04d}_"
            f"target_{safe_name(target_name)}_"
            f"pred_{safe_name(prediction_name)}_"
            f"{status}"
        )

        png_path = args.output_dir / f"{filename}.png"
        pdf_path = args.output_dir / f"{filename}.pdf"

        figure.savefig(
            png_path,
            dpi=200,
            bbox_inches="tight",
        )
        figure.savefig(
            pdf_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"Saved: {png_path}")
        print(
            f"  target={target_name}, "
            f"prediction={prediction_name}, "
            f"probability={probability:.3f}, "
            f"status={status}"
        )


if __name__ == "__main__":
    main()
