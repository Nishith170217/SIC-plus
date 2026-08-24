# SIC training for Oxford-IIIT Pet.
# 37-class single-label classification with B-cos ResNet50.

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from bcos import BcosEncoderWrapper, resnet50_long
from bcos.pretrained_imagenet import densenet121_long
from pets_dataset import N_PET_CLASSES, get_pets_dataloader
from sic import SIC
from train_sic_voc import get_optimizer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SIC on Oxford-IIIT Pet."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/pets"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_way", type=int, default=30)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "densenet121"],
        default="resnet50",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--support_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=32,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.n_shot < 1:
        parser.error("--n_shot must be at least 1")
    if args.accumulation_steps < 1:
        parser.error(
            "--accumulation_steps must be at least 1"
        )
    if args.n_way is not None:
        if not 1 <= args.n_way <= N_PET_CLASSES:
            parser.error(
                f"--n_way must be between 1 and "
                f"{N_PET_CLASSES}"
            )

    return args


def one_hot_targets(targets):
    return functional.one_hot(
        targets.long(),
        num_classes=N_PET_CLASSES,
    ).float()


def limited_batches(loader, maximum):
    if maximum is None:
        return len(loader)
    return min(len(loader), maximum)


def validate(
    model,
    validation_loader,
    criterion,
    device,
    epoch,
    epochs,
    maximum_batches=None,
):
    model.eval()

    print("Precomputing validation supports...")

    with torch.no_grad():
        model.precompute()

    total_loss = 0.0
    total_images = 0
    all_logits = []
    all_targets = []

    number_of_batches = limited_batches(
        validation_loader,
        maximum_batches,
    )

    with torch.inference_mode():
        progress = tqdm(
            validation_loader,
            total=number_of_batches,
            desc=f"Validate {epoch}/{epochs}",
        )

        for batch_index, (images, targets) in enumerate(
            progress
        ):
            if batch_index >= number_of_batches:
                break

            images = images.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True,
            )

            logits = model.predict(images)
            loss = criterion(
                logits,
                one_hot_targets(targets),
            )

            batch_size = len(images)
            total_loss += float(loss) * batch_size
            total_images += batch_size

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    predictions = logits.argmax(dim=1)
    top_five = logits.topk(k=5, dim=1).indices

    top1_accuracy = float(
        (predictions == targets).float().mean() * 100
    )
    top5_accuracy = float(
        (top_five == targets.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        * 100
    )

    targets_numpy = targets.numpy()
    predictions_numpy = predictions.numpy()

    macro_f1 = float(
        f1_score(
            targets_numpy,
            predictions_numpy,
            average="macro",
            zero_division=0,
        )
        * 100
    )
    balanced_accuracy = float(
        balanced_accuracy_score(
            targets_numpy,
            predictions_numpy,
        )
        * 100
    )

    return {
        "loss": total_loss / total_images,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_top1,
    metrics,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_top1": best_top1,
            "top1_accuracy": metrics["top1_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics[
                "balanced_accuracy"
            ],
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = args.amp and device.type == "cuda"

    print(f"Device: {device}")
    print(f"Mixed precision: {amp_enabled}")

    (
        train_loader,
        validation_loader,
        support_loader,
    ) = get_pets_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=args.val_batch_size,
    )

    train_batches = limited_batches(
        train_loader,
        args.max_train_batches,
    )
    validation_batches = limited_batches(
        validation_loader,
        args.max_val_batches,
    )

    print(f"Training batches per epoch: {train_batches}")
    print(
        f"Validation batches per epoch: "
        f"{validation_batches}"
    )

    if args.backbone == "resnet50":
        backbone = resnet50_long(pretrained=True)
    else:
        backbone = densenet121_long(pretrained=True)

    print(f"Backbone: {args.backbone}")

    featurizer = BcosEncoderWrapper(backbone)

    model = SIC(
        featurizer=featurizer,
        n_classes=N_PET_CLASSES,
        proj_dim=128,
        n_way=args.n_way,
        n_shot=args.n_shot,
        temperature=10,
        support_loader=support_loader,
        device=device,
        multilabel=False,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer_steps_per_epoch = math.ceil(
        train_batches / args.accumulation_steps
    )

    optimizer, scheduler = get_optimizer(
    [p for p in model.parameters() if p.requires_grad],
    args.lr,
    args.epochs,
    optimizer_steps_per_epoch,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    checkpoint_path = (
        args.results_dir / "checkpoint.pth"
    )
    best_model_path = (
        args.results_dir / "best_model.pth"
    )
    final_model_path = (
        args.results_dir / "sic_pets_final.pth"
    )
    history_path = args.results_dir / "history.json"

    start_epoch = 0
    best_top1 = float("-inf")
    history = []

    if args.resume and checkpoint_path.is_file():
        print(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
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

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1
        best_top1 = checkpoint.get(
            "best_top1",
            float("-inf"),
        )

        if history_path.is_file():
            with open(history_path) as file:
                history = json.load(file)

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        processed_batches = 0

        progress = tqdm(
            train_loader,
            total=train_batches,
            desc=f"Train {epoch_number}/{args.epochs}",
        )

        for batch_index, (images, targets) in enumerate(
            progress
        ):
            if batch_index >= train_batches:
                break

            images = images.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images, targets)
                loss = criterion(
                    logits,
                    one_hot_targets(targets),
                )
                backward_loss = (
                    loss / args.accumulation_steps
                )

            scaler.scale(backward_loss).backward()

            is_accumulation_boundary = (
                (batch_index + 1)
                % args.accumulation_steps
                == 0
            )
            is_last_batch = (
                batch_index + 1 == train_batches
            )

            if (
                is_accumulation_boundary
                or is_last_batch
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += float(loss)
            processed_batches += 1

        train_loss = running_loss / processed_batches

        metrics = validate(
            model,
            validation_loader,
            criterion,
            device,
            epoch_number,
            args.epochs,
            maximum_batches=args.max_val_batches,
        )

        print(
            f"Epoch {epoch_number} train loss: "
            f"{train_loss:.4f}"
        )
        print(
            f"Epoch {epoch_number} validation: "
            f"loss={metrics['loss']:.4f}, "
            f"top1={metrics['top1_accuracy']:.2f}%, "
            f"top5={metrics['top5_accuracy']:.2f}%, "
            f"macro_F1={metrics['macro_f1']:.2f}%, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.2f}%"
        )

        if metrics["top1_accuracy"] > best_top1:
            best_top1 = metrics["top1_accuracy"]
            torch.save(
                model.state_dict(),
                best_model_path,
            )

        save_checkpoint(
            checkpoint_path,
            epoch_index,
            model,
            optimizer,
            scheduler,
            scaler,
            best_top1,
            metrics,
            args,
        )

        history.append(
            {
                "epoch": epoch_number,
                "train_loss": train_loss,
                **metrics,
            }
        )

        with open(history_path, "w") as file:
            json.dump(history, file, indent=2)

    torch.save(model.state_dict(), final_model_path)

    print(f"Best Top-1 accuracy: {best_top1:.2f}%")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
