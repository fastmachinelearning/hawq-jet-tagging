import logging
import argparse

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from utils.train_utils import *

parser = argparse.ArgumentParser(description="HAWQ Jet-Tagging")
parser.add_argument("--data", metavar="DIR", help="Path to dataset.")
parser.add_argument(
    "--config",
    type=str,
    default="config/config_6bit.yml",
    help="Bit configuration file for model.",
)
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N", help="Number of total epochs to run."
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="Initial learning rate.",
    dest="lr",
)
parser.add_argument(
    "--save-path",
    type=str,
    default="checkpoints/",
    help="path to save the quantized model",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1024,
    type=int,
    metavar="N",
    help="Mini-batch size (default: 1024), this is the total.",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="Print frequency (default: 10)",
)
parser.add_argument(
    "--batch-norm",
    action="store_true",
    help="Implement MLP with batch normalization.",
)
parser.add_argument(
    "--l1",
    action="store_true",
    help="Implement L1 regularization.",
)
parser.add_argument(
    "--l2",
    action="store_true",
    help="Implement L2 regularization.",
)
args = parser.parse_args()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    setup_logger(args)
    train_loader, val_loader = load_dataset(args.data, args.batch_size, args.config)

    model = load_model(args)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.BCELoss()

    best_epoch = 0
    best_acc = 0
    loss_record = list()
    acc_record = list()

    for epoch in range(args.epochs):
        # train for one epoch
        epoch_loss = train(
            train_loader, model, criterion, optimizer, epoch, scheduler, args
        )
        acc = validate(val_loader, model, criterion, args)
        logging.info(f"lr: {optimizer.param_groups[0]['lr']}")

        # record loss and accuracy
        loss_record.append(epoch_loss)
        acc_record.append(acc)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        best_epoch = epoch if is_best == True else best_epoch

        logging.info(f"Best acc at epoch {best_epoch+1}: {best_acc}")
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.save_path,
            )

    log_training(loss_record, "model_loss.json", args.save_path)
    log_training(acc_record, "model_acc.json", args.save_path)


# python train_jettagger.py --epochs 5 --lr 0.01 --data /data1/jcampos/datasets --batch-size 1024 --save-path checkpoints  --config config/config_6bit.yml
if __name__ == "__main__":

    model = main()
