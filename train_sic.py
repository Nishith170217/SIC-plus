# @author Tom Nuno Wolf, Technical University of Munich
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import torch
import numpy as np
from tqdm import tqdm
from dogs_dataset import get_dogs_dataloader
from functools import partial
from sic import SIC
from bcos import resnet50_long, BcosEncoderWrapper
from torch.nn.functional import one_hot
from pathlib import Path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def default_to_one_hot(x, num_classes):
    targets = one_hot(x.long(), num_classes)
    return targets.float()

def get_optimizer(params, lr, epochs, n_iters_per_epoch):
    optimizer = torch.optim.AdamW(
        params=params, lr=lr, weight_decay=0.
    )
    n_iters_warmup = int(2 * n_iters_per_epoch)
    start_decay = int(0.4 * n_iters_per_epoch * epochs)
    step_counts = int(0.1 * n_iters_per_epoch * epochs)
    milestones = [n_iters_warmup, start_decay]

    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_iters_warmup)
    scheduler2 = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=start_decay,
    )
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_counts, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3],
                                                        milestones=milestones, verbose=True)
    print("Milestones....", milestones)
    return optimizer, scheduler

def default_step(model, x, y, to_one_hot, criterion):
    logits = model(x, y)
    if logits.size(-1) == 1:
        logits = logits.squeeze()
    preds = (logits > 0).float()
    targets = to_one_hot(y)

    loss = criterion(logits, targets)

    return loss, preds, targets

def default_val_step(model, x, y, to_one_hot, criterion):
    assert model.training == False
    logits = model.predict(x)  # must use model.predict() for inference!!!
    if logits.size(-1) == 1:
        logits = logits.squeeze()
    preds = torch.argmax(logits, dim=1)

    loss = criterion(logits, to_one_hot(y))

    return loss, preds, y

def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--n_way", type=int, default=None, help="Reduces number of classes sampled for training on small GPU memory to n_way classes")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results into")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    args = parser.parse_args(args=args)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_classes = 120
    train_loader, val_loader, support_loader = get_dogs_dataloader(args.data_dir, batch_size=args.batch_size, num_workers=8, is_bcos=True, support_batch_size=4, val_batch_size=1)

    # initialize featurizer
    featurizer = BcosEncoderWrapper(resnet50_long(pretrained=True))

    # initialize model
    sic = SIC(
        featurizer=featurizer,
        n_classes=n_classes,
        proj_dim=128,
        n_way=args.n_way,  # use if number of classes is too large to sample from all classes during training
        n_shot=3,
        temperature=10,
        support_loader=support_loader,
        device=device,
    )
    # sanity check
        # sanity check
         # sanity check
    sic.eval()
    with torch.no_grad():
        sic.precompute()

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=None)  # add pos_weight if required
    to_one_hot = partial(default_to_one_hot, num_classes=n_classes)
    optimizer, scheduler = get_optimizer(
        [p for p in sic.parameters() if p.requires_grad],
        lr=0.001,
        epochs=args.epochs,
        n_iters_per_epoch=len(train_loader),
    )

    start_epoch = 0
    # Resume from checkpoint if exists
    checkpoint_path = Path(args.results_dir) / "checkpoint.pth"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        sic.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
    
        # TRAIN
        total_loss = 0
        sic.train()
        optimizer.zero_grad()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            loss, preds, targets = default_step(sic, x.to(device), y.to(device), to_one_hot, criterion)
            loss = loss / args.accumulation_steps
            loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.accumulation_steps
        print(f"Epoch {epoch} train loss: {total_loss / len(train_loader)}")

        # VALIDATE
        total_loss = 0

        ##########
        # validation routine.
        #
        # Must first set to eval, followed by precomputing support vectors.
        # Then is ready for evaluation.
        ##########
        sic.eval()

        with torch.no_grad():
            sic.precompute()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for (x, y) in tqdm(val_loader):
                loss, preds, targets = default_val_step(
                    sic,
                    x.to(device),
                    y.to(device),
                    to_one_hot,
                    criterion,
                )

                all_preds.append(preds.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
                total_loss += loss.item()

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # compute foreground classes accuracy only
        accuracy = (all_preds == all_targets).mean() * 100

        print(
            f"Epoch {epoch} val loss: {total_loss / len(val_loader)}, "
            f"val accuracy: {accuracy:.2f}%"
        )

        torch.cuda.empty_cache()
        # Save checkpoint (inside epoch loop)
        torch.save({
            'epoch': epoch,
            'model_state_dict': sic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    # After loop ends - save final model
    torch.save(sic.state_dict(), Path(args.results_dir) / "sic_dogs.pth")


if __name__ == "__main__":
    main()
