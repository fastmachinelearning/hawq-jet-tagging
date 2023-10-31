'''
TinyMLPerf benchmark with Hessian-AWare Quantization  
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

train.py desc: loads data, trains and saves model, plots training metrics
'''

import datetime

EPOCHS = 500
BS = 32

# get date and time to save model
dt = datetime.datetime.today()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute
second = dt.second

experiment_name = f"{month}.{day}.{year}-{hour}.{minute}.{second}"


import os
import time
import torch
import torchinfo
import numpy as np
import pytorch_lightning as pl 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from model import RN08
from data import get_test_dataloader, get_training_dataloader


def main(args):
    # ------------------------
    # 0 PREPARE EXPERIMENT
    # ------------------------
    start_time = time.time()

    checkpoint_dir = f'{os.environ["HAWQ_JET_TAGGING"]}/checkpoints/ic'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # ------------------------
    # 1 PREPARE DATA
    # ------------------------
    data_dir = f'{os.environ["HAWQ_JET_TAGGING"]}/data/cifar10'

    train_dataloader = get_training_dataloader(
        data_dir,
        batch_size=32,
        num_workers=2,
        shuffle=True,
    )

    test_dataloader = get_test_dataloader(
        data_dir,
        batch_size=32,
        num_workers=2,
        shuffle=False,
    )
    
    # ------------------------
    # 2 INIT LIGHTNING MODEL
    # ------------------------
    model = RN08(
        precision=[],
        lr=args.lr,
    )

    torchinfo.summary(model, input_size=(1, 3, 32, 32))  # (B, C, H, W)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=checkpoint_dir, name=experiment_name)
    
    # Stop training when model converges
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, verbose=True, mode="min")

    # Save top-1 checkpoints based on Val/Loss
    top3_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(checkpoint_dir, experiment_name),
        filename=f'resnet_best',
        auto_insert_metric_name=False,
    )
    print(f'Saving to dir: {os.path.join(checkpoint_dir, experiment_name)}')
    print(f'Running experiment: {experiment_name}')

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=tb_logger,
        callbacks=[top3_checkpoint_callback, early_stop_callback],
    )

    # ------------------------
    # 4 TRAIN MODEL
    # ------------------------
    if args.train:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # ------------------------
    # 5 EVALUTE MODEL
    # ------------------------
    if args.train or args.evaluate:
        checkpoint_file = os.path.join(checkpoint_dir, experiment_name, 'resnet_best.ckpt')
        print('Loading checkpoint:', checkpoint_file)
        model.load_state_dict(torch.load(checkpoint_file)['state_dict'])

        metrics = trainer.test(model=model, dataloaders=test_dataloader)[0]

        elapsed_time = time.time() - start_time

        with open(os.path.join(checkpoint_dir, "summary.txt"), "a") as f:
            f.write("=====================================================\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"\tTest Loss: {metrics['test_loss']}\n")
            f.write(f"\tTest Acc: {metrics['test_acc']}\n")
            f.write(f"\tExecution Time (s): {elapsed_time}\n")
            f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    # todo: require a 'checkpoint' option when evaluating 
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "auto"], default="auto"
    )
    args = parser.parse_args()
    main(args)
