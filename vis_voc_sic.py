# SIC visualization for Pascal VOC 2007.

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from sic import SIC
from voc_dataset import VOC_CLASSES, get_voc_dataloader


CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(VOC_CLASSES)
}


def prediction_status(target, probability, threshold):
    predicted = probability >= threshold

    if target == -1:
        return "difficult"

    if target == 1 and predicted:
        return "true_positive"

    if target == 1 and not predicted:
        return "false_negative"

    if target == 0 and predicted:
        return "false_positive"

    return "true_negative"


def select_multilabel_sample(
    model,
    dataset,
    device,
    threshold,
):
    for sample_index in range(len(dataset)):
        image, target = dataset[sample_index]
        positive_classes = torch.where(target == 1)[0]

        if len(positive_classes) < 2:
            continue

        with torch.inference_mode():
            logits = model.predict(
                image.unsqueeze(0).to(device)
            )
            probabilities = torch.sigmoid(logits)[0].cpu()

        correctly_predicted = (
            probabilities[positive_classes] >= threshold
        ).sum()

        if correctly_predicted >= 2:
            return (
                sample_index,
                image,
                target,
                probabilities,
            )

    raise RuntimeError(
        "Could not find a correctly predicted multi-label sample"
    )


def load_sample(
    model,
    dataset,
    device,
    threshold,
    sample_index,
):
    if sample_index is None:
        return select_multilabel_sample(
            model,
            dataset,
            device,
            threshold,
        )

    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"Sample index {sample_index} is outside "
            f"0..{len(dataset) - 1}"
        )

    image, target = dataset[sample_index]

    with torch.inference_mode():
        logits = model.predict(
            image.unsqueeze(0).to(device)
        )
        probabilities = torch.sigmoid(logits)[0].cpu()

    return (
        sample_index,
        image,
        target,
        probabilities,
    )


def choose_classes(
    target,
    probabilities,
    threshold,
    requested_classes,
    max_classes,
):
    if requested_classes:
        unknown = [
            name
            for name in requested_classes
            if name not in CLASS_TO_INDEX
        ]

        if unknown:
            raise ValueError(
                f"Unknown VOC classes: {unknown}"
            )

        return [
            CLASS_TO_INDEX[name]
            for name in requested_classes
        ]

    positive_classes = torch.where(target == 1)[0].tolist()

    false_positives = torch.where(
        (target == 0)
        & (probabilities >= threshold)
    )[0].tolist()

    selected = positive_classes + false_positives

    return selected[:max_classes]


def plot_class_explanation(
    model,
    image,
    target,
    probabilities,
    class_index,
    threshold,
    percentile,
    smooth,
):
    number_of_columns = model.n_shot + 2

    figure, axes = plt.subplots(
        2,
        number_of_columns,
        figsize=(2.5 * number_of_columns, 5.2),
        constrained_layout=True,
    )

    class_name = VOC_CLASSES[class_index]
    probability = float(probabilities[class_index])
    target_value = int(target[class_index])
    status = prediction_status(
        target_value,
        probability,
        threshold,
    )

    axes[0, 0].imshow(
        image[:3].permute(1, 2, 0)
    )
    axes[0, 0].set_title("Test image")
    axes[0, 0].axis("off")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.95,
        f"Class: {class_name}\n"
        f"Probability: {probability:.3f}\n"
        f"Target: {target_value}\n"
        f"Status: {status}",
        va="top",
        fontsize=10,
    )

    model.plot_calibration_scores(
        image.unsqueeze(0).to(model.device),
        class_index,
        axes[0, 1],
        color=(
            "tab:blue"
            if status == "true_positive"
            else "tab:red"
        ),
        y_probs=False,
    )
    axes[0, 1].set_title("Support evidence")
    axes[0, 1].set_xticks(range(model.n_shot + 2))
    axes[0, 1].set_xticklabels(
        (
            ["Total"]
            + [
                f"S{support_number + 1}"
                for support_number
                in range(model.n_shot)
            ]
            + ["Bias"]
        ),
        rotation=0,
        fontsize=8,
    )
    axes[0, 1].tick_params(
        axis="x",
        direction="out",
        pad=3,
    )
    test_explanation = model.explain_prediction(
        image.unsqueeze(0).to(model.device),
        class_index,
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
            class_index * model.n_shot
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
            f"Support {support_number + 1}"
        )
        axes[0, column].axis("off")

        axes[1, column].imshow(
            prototype["explanation"]
        )
        axes[1, column].set_title(
            f"Support contribution {support_number + 1}"
        )
        axes[1, column].axis("off")

    figure.suptitle(
        f"Pascal VOC SIC explanation: {class_name}",
        fontsize=14,
    )

    return figure, status


def parse_args():
    parser = argparse.ArgumentParser()

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
        default=Path("visualizations/voc"),
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=9,
    )

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if args.max_classes < 1:
        parser.error("--max_classes must be at least 1")

    if args.n_shot < 1:
        parser.error("--n_shot must be at least 1")

    return args


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, val_loader, support_loader = get_voc_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
    )

    backbone = densenet121_long(pretrained=False)
    featurizer = BcosEncoderWrapper(backbone)

    model = SIC(
        featurizer=featurizer,
        n_classes=len(VOC_CLASSES),
        proj_dim=128,
        n_way=None,
        n_shot=args.n_shot,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=True,
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Precomputing support vectors...")
    model.precompute()

    (
        sample_index,
        image,
        target,
        probabilities,
    ) = load_sample(
        model,
        val_loader.dataset,
        device,
        args.threshold,
        args.sample_index,
    )

    selected_classes = choose_classes(
        target,
        probabilities,
        args.threshold,
        args.classes,
        args.max_classes,
    )

    if not selected_classes:
        raise RuntimeError(
            "No classes selected for visualization"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Sample index: {sample_index}")
    print(
        "Ground-truth classes:",
        [
            VOC_CLASSES[index]
            for index in torch.where(target == 1)[0]
        ],
    )

    for class_index in selected_classes:
        figure, status = plot_class_explanation(
            model,
            image,
            target,
            probabilities,
            class_index,
            args.threshold,
            args.percentile,
            args.smooth,
        )

        class_name = VOC_CLASSES[class_index]
        filename = (
            f"sample_{sample_index:04d}_"
            f"nshot_{args.n_shot}_"
            f"{class_name}_{status}"
        )

        figure.savefig(
            args.output_dir / f"{filename}.png",
            dpi=200,
        )
        figure.savefig(
            args.output_dir / f"{filename}.pdf",
            dpi=300,
        )
        plt.close(figure)

        print(
            f"Saved {class_name}: "
            f"probability={probabilities[class_index]:.3f}, "
            f"status={status}"
        )


if __name__ == "__main__":
    main()
