# SIC training script for Pascal VOC 2007
# Multi-label classification with B-cos DenseNet121

import argparse
import math
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from bcos import BcosEncoderWrapper
from bcos.pretrained_imagenet import densenet121_long
from sic import SIC
from voc_dataset import VOC_CLASSES, get_voc_dataloader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(params, lr, epochs, optimizer_steps_per_epoch):
    optimizer = torch.optim.AdamW(
        params=params,
        lr=lr,
        weight_decay=0.0,
    )

    total_steps = max(
        1,
        epochs * optimizer_steps_per_epoch,
    )
    warmup_steps = min(
        2 * optimizer_steps_per_epoch,
        total_steps,
    )
    decay_start = max(
        warmup_steps,
        int(0.4 * total_steps),
    )
    decay_interval = max(
        1,
        int(0.1 * total_steps),
    )

    def lr_factor(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / warmup_steps

        if step < decay_start:
            return 1.0

        decay_count = (
            step - decay_start
        ) // decay_interval + 1

        return 0.5 ** decay_count

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )

    print(
        f"LR schedule: total={total_steps}, "
        f"warmup={warmup_steps}, "
        f"decay_start={decay_start}, "
        f"decay_interval={decay_interval}"
    )

    return optimizer, scheduler


def masked_bce_loss(logits, targets, criterion):
    """
    Target encoding:
        1 = positive
        0 = negative
       -1 = difficult; ignored
    """
    valid_mask = targets >= 0

    if not valid_mask.any():
        raise ValueError("Batch contains no valid VOC labels")

    binary_targets = targets.clamp_min(0)
    element_losses = criterion(logits, binary_targets)

    return element_losses[valid_mask].mean()


def train_step(model, images, targets, criterion):
    logits = model(images, targets)
    return masked_bce_loss(logits, targets, criterion)


def val_step(model, images, targets, criterion):
    if model.training:
        raise RuntimeError(
            "Model must be in evaluation mode during validation"
        )

    logits = model.predict(images)
    loss = masked_bce_loss(logits, targets, criterion)
    probabilities = torch.sigmoid(logits)

    return loss, probabilities


def compute_map(all_scores, all_targets):
    all_scores = np.asarray(all_scores)
    all_targets = np.asarray(all_targets)

    if all_scores.shape != all_targets.shape:
        raise ValueError(
            "Scores and targets must have identical shapes, "
            f"got {all_scores.shape} and {all_targets.shape}"
        )

    if all_targets.ndim != 2:
        raise ValueError(
            "Expected scores and targets shaped [samples, classes]"
        )

    per_class_ap = []

    for class_idx in range(all_targets.shape[1]):
        class_targets = all_targets[:, class_idx]
        class_scores = all_scores[:, class_idx]

        valid_mask = class_targets >= 0
        class_targets = class_targets[valid_mask]
        class_scores = class_scores[valid_mask]

        if class_targets.sum() == 0:
            per_class_ap.append(np.nan)
            continue

        per_class_ap.append(
            average_precision_score(
                class_targets,
                class_scores,
            )
        )

    per_class_ap = np.asarray(
        per_class_ap,
        dtype=np.float64,
    )

    return (
        np.nanmean(per_class_ap) * 100,
        per_class_ap * 100,
    )


def compute_f1(all_scores, all_targets, threshold=0.5):
    all_scores = np.asarray(all_scores)
    all_targets = np.asarray(all_targets)

    if all_scores.shape != all_targets.shape:
        raise ValueError(
            "Scores and targets must have identical shapes"
        )

    predictions = (
        all_scores >= threshold
    ).astype(np.int64)

    valid_mask = all_targets >= 0

    micro_f1 = f1_score(
        all_targets[valid_mask].astype(np.int64),
        predictions[valid_mask],
        average="binary",
        zero_division=0,
    )

    per_class_f1 = []

    for class_idx in range(all_targets.shape[1]):
        class_valid = valid_mask[:, class_idx]

        class_targets = all_targets[
            class_valid,
            class_idx,
        ].astype(np.int64)

        class_predictions = predictions[
            class_valid,
            class_idx,
        ]

        per_class_f1.append(
            f1_score(
                class_targets,
                class_predictions,
                average="binary",
                zero_division=0,
            )
        )

    return (
        micro_f1 * 100,
        np.mean(per_class_f1) * 100,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/voc",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--n_way",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--support_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
    )

    parsed = parser.parse_args(args=args)

    if parsed.epochs < 1:
        parser.error("--epochs must be at least 1")

    if parsed.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    if parsed.lr <= 0:
        parser.error("--lr must be greater than zero")

    if parsed.accumulation_steps < 1:
        parser.error(
            "--accumulation_steps must be at least 1"
        )

    if parsed.num_workers < 0:
        parser.error("--num_workers cannot be negative")

    if parsed.n_shot < 1:
        parser.error("--n_shot must be at least 1")

    if parsed.support_batch_size < 1:
        parser.error(
            "--support_batch_size must be at least 1"
        )

    if parsed.val_batch_size < 1:
        parser.error(
            "--val_batch_size must be at least 1"
        )

    if (
        parsed.max_train_batches is not None
        and parsed.max_train_batches < 1
    ):
        parser.error(
            "--max_train_batches must be at least 1"
        )

    if (
        parsed.max_val_batches is not None
        and parsed.max_val_batches < 1
    ):
        parser.error(
            "--max_val_batches must be at least 1"
        )

    if not 0 < parsed.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    return parsed


def main(args=None):
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )

    args = parse_args(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = args.amp and device.type == "cuda"

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    train_loader, val_loader, support_loader = (
        get_voc_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_bcos=True,
            support_batch_size=args.support_batch_size,
            val_batch_size=args.val_batch_size,
        )
    )

    train_batches_per_epoch = len(train_loader)
    val_batches_per_epoch = len(val_loader)

    if args.max_train_batches is not None:
        train_batches_per_epoch = min(
            train_batches_per_epoch,
            args.max_train_batches,
        )

    if args.max_val_batches is not None:
        val_batches_per_epoch = min(
            val_batches_per_epoch,
            args.max_val_batches,
        )

    print(
        f"Training batches per epoch: "
        f"{train_batches_per_epoch}"
    )
    print(
        f"Validation batches per epoch: "
        f"{val_batches_per_epoch}"
    )

    backbone = densenet121_long(pretrained=True)
    featurizer = BcosEncoderWrapper(backbone)

    model = SIC(
        featurizer=featurizer,
        n_classes=len(VOC_CLASSES),
        proj_dim=128,
        n_way=args.n_way,
        n_shot=args.n_shot,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=True,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="none"
    )

    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch
        / args.accumulation_steps
    )

    optimizer, scheduler = get_optimizer(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.lr,
        epochs=args.epochs,
        optimizer_steps_per_epoch=(
            optimizer_steps_per_epoch
        ),
    )

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=use_amp,
    )

    checkpoint_path = results_dir / "checkpoint.pth"
    best_model_path = results_dir / "best_model.pth"
    start_epoch = 0
    best_map = float("-inf")

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint.get(
            "best_map",
            float("-inf"),
        )

        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        num_batches = train_batches_per_epoch

        final_group_size = (
            num_batches % args.accumulation_steps
            or args.accumulation_steps
        )

        train_iterator = islice(
            train_loader,
            num_batches,
        )

        for batch_idx, (images, targets) in enumerate(
            tqdm(
                train_iterator,
                total=num_batches,
                desc=f"Train {epoch + 1}/{args.epochs}",
            )
        ):
            images = images.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type='cuda',
                enabled=use_amp,
            ):
                batch_loss = train_step(
                    model,
                    images,
                    targets,
                    criterion,
                )

            in_final_group = (
                batch_idx
                >= num_batches - final_group_size
            )

            loss_divisor = (
                final_group_size
                if in_final_group
                else args.accumulation_steps
            )

            scaler.scale(
                batch_loss / loss_divisor
            ).backward()

            accumulation_complete = (
                (batch_idx + 1)
                % args.accumulation_steps
                == 0
            )
            final_batch = (
                batch_idx + 1 == num_batches
            )

            if accumulation_complete or final_batch:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

            train_loss += batch_loss.item()

        train_loss /= num_batches

        print(
            f"Epoch {epoch + 1} train loss: "
            f"{train_loss:.4f}"
        )

        model.eval()
        model.precompute()

        val_scores = []
        val_targets = []
        val_loss = 0.0

        val_iterator = islice(
            val_loader,
            val_batches_per_epoch,
        )

        with torch.inference_mode():
            for images, targets in tqdm(
                val_iterator,
                total=val_batches_per_epoch,
                desc=f"Validate {epoch + 1}/{args.epochs}",
            ):
                images = images.to(
                    device,
                    non_blocking=True,
                )
                targets = targets.to(
                    device,
                    non_blocking=True,
                )

                loss, probabilities = val_step(
                    model,
                    images,
                    targets,
                    criterion,
                )

                val_loss += loss.item()
                val_scores.append(
                    probabilities.cpu().numpy()
                )
                val_targets.append(
                    targets.cpu().numpy()
                )

        val_scores = np.concatenate(
            val_scores,
            axis=0,
        )
        val_targets = np.concatenate(
            val_targets,
            axis=0,
        )
        val_loss /= val_batches_per_epoch

        map_score, per_class_ap = compute_map(
            val_scores,
            val_targets,
        )
        micro_f1, macro_f1 = compute_f1(
            val_scores,
            val_targets,
            threshold=args.threshold,
        )

        print(
            f"Epoch {epoch + 1} validation: "
            f"loss={val_loss:.4f}, "
            f"mAP={map_score:.2f}%, "
            f"micro_F1={micro_f1:.2f}%, "
            f"macro_F1={macro_f1:.2f}%"
        )

        for class_name, class_ap in zip(
            VOC_CLASSES,
            per_class_ap,
        ):
            print(
                f"  {class_name:12s}: "
                f"AP={class_ap:.2f}%"
            )

        is_best = map_score > best_map

        if is_best:
            best_map = map_score
            torch.save(
                model.state_dict(),
                best_model_path,
            )

        checkpoint = {
            "epoch": epoch,
            "best_map": best_map,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
            "map": map_score,
            "per_class_ap": per_class_ap,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(
        model.state_dict(),
        results_dir / "sic_voc_final.pth",
    )


if __name__ == "__main__":
    main()
