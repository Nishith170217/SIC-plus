import argparse
from pathlib import Path

import torch

from eval_pets import get_class_names, load_model
from pets_dataset import get_pets_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    args.n_shot = 3
    args.support_batch_size = 32

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, validation_loader, support_loader = get_pets_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_bcos=True,
        support_batch_size=args.support_batch_size,
        val_batch_size=args.batch_size,
    )

    class_names = get_class_names(args.data_dir)
    model = load_model(args, support_loader, device)

    with torch.no_grad():
        model.precompute()

    found = 0
    dataset_index = 0

    with torch.no_grad():
        for images, targets in validation_loader:
            logits = model.predict(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            probabilities = torch.sigmoid(logits).cpu()

            for offset in range(len(targets)):
                target = int(targets[offset])
                prediction = int(predictions[offset])

                if target != prediction:
                    confidence = float(
                        probabilities[offset, prediction]
                    )

                    print(
                        f"index={dataset_index + offset} "
                        f"target={class_names[target]} "
                        f"prediction={class_names[prediction]} "
                        f"confidence={confidence:.3f}"
                    )

                    found += 1

                    if found >= args.limit:
                        return

            dataset_index += len(targets)


if __name__ == "__main__":
    main()
