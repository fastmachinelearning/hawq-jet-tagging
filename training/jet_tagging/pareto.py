import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from config import config_dict
from train_utils import *
from utilities.calc_bops import calc_hawq_bops
from utilities.neural_eff import calc_neural_efficieny


# ------------------------------------------------------------
# setup and training
# ------------------------------------------------------------
def main(args):
    global_save_path = os.path.join(os.environ["HAWQ_JET_TAGGING"], "checkpoints/jets/pareto")
    args.config = os.path.join(os.environ["HAWQ_JET_TAGGING"], args.config)
    train_loader, val_loader = load_dataset(os.path.join(os.environ["HAWQ_JET_TAGGING"], "data/jets"), args.batch_size, args.config)

    for key, cfg in config_dict.items():
        print(f"Starting {key}")
        args.save_path = os.path.join(global_save_path, key)
        setup_logger(args)

        model = load_model(args)
        config_model(model, cfg)
        logging.info(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
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

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.save_path,
            )
            logging.info(f"Best acc at epoch {best_epoch}: {best_acc}")
            if is_best:
                total_bops, percent_pruned = calc_hawq_bops(
                    model, model.quant_input.activation_bit
                )
                neural_efficiency = calc_neural_efficieny(model, val_loader, args)

                log_training(
                    [best_acc, total_bops, percent_pruned, neural_efficiency],
                    "metrics.json",
                    args.save_path,
                )

        log_training(loss_record, "model_loss.json", args.save_path)
        log_training(acc_record, "model_acc.json", args.save_path)
        logging.info(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAWQ Jet-Tagging")
    parser.add_argument("--data", metavar="DIR", help="Path to dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/config_6bit.yml",
        help="Bit and dataset configuration.",
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
        help="Path to save the quantized model.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1024,
        type=int,
        metavar="N",
        help="Mini-batch size (default: 1024).",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=200,
        type=int,
        metavar="N",
        help="Print frequency (default: 200)",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Implement with batch normalization.",
    )
    parser.add_argument(
        "--batch-norm-fold",
        action="store_true",
        help="Implement with batch normalization folding.",
    )
    parser.add_argument(
        "--dropout",
        action="store_true",
        help="Train with dropout (Default=0.2).",
    )
    parser.add_argument(
        "--l1",
        action="store_true",
        help="Implement L1 regularization.",
    )
    parser.add_argument(
        "--l1-lambda",
        default=1e-7,
        type=float,
        help="L1 regularization rate.",
    )
    parser.add_argument(
        "--l2",
        action="store_true",
        help="Implement L2 regularization.",
    )
    parser.add_argument(
        "--l2-lambda",
        default=1e-7,
        type=float,
        help="L2 regularization rate.",
    )
    args = parser.parse_args()

    main(args)
