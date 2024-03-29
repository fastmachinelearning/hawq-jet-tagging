import os
import sys
import time
import datetime
from tqdm import tqdm
import torch
import torchinfo
import numpy as np
import multiprocessing
import pytorch_lightning as pl 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from pl_trainer import AutoEncoder
from autoencoder_datamodule import AutoEncoderDataModule
from utils_pt import unnormalize, emd


def timeStamp():
    # get date and time to save model
    dt = datetime.datetime.today()
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    experiment_name = f"{month}.{day}.{year}-{hour}.{minute}.{second}"
    return experiment_name


def test_model(model, test_loader):
    """
    Our own testing loop instead of using the trainer.test() method so that we
    can multithread EMD computation on the CPU
    """
    model.eval()
    input_calQ_list = []
    output_calQ_list = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(model.device)
            output = model(x)
            input_calQ = model.map_to_calq(x)
            output_calQ_fr = model.map_to_calq(output)
            input_calQ = torch.stack(
                [input_calQ[i] * model.val_sum[i] for i in range(len(input_calQ))]
            )  # shape = (batch_size, 48)
            output_calQ = unnormalize(
                torch.clone(output_calQ_fr), model.val_sum
            )  # ae_out
            input_calQ_list.append(input_calQ)
            output_calQ_list.append(output_calQ)
    input_calQ = np.concatenate([i_calQ.cpu() for i_calQ in input_calQ_list], axis=0)
    output_calQ = np.concatenate([o_calQ.cpu() for o_calQ in output_calQ_list], axis=0)
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        emd_list = pool.starmap(emd, zip(input_calQ, output_calQ))
    print(f"EMD computation time: {time.time() - start_time} seconds")
    average_emd = np.mean(np.array(emd_list))
    print(f"Average EMD: {average_emd}")
    return average_emd


def main(args):
    start_time = time.time()
    experiment_name = timeStamp()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # ------------------------
    # 0 PREPARE DATA
    # ------------------------
    data_module = AutoEncoderDataModule.from_argparse_args(args)
    data_module.train_bs = 32  # I DON'T LIKE TOO HARDCODED
    data_module.test_bs = 32  # I DON'T LIKE TOO HARDCODED
    if args.process_data:
        print("Processing data...")
        data_module.process_data()
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = AutoEncoder(
        accelerator=args.accelerator, 
        quantize=args.quantize,
        precision=args.bitwidth,
        learning_rate=1e-3,
        econ_type="baseline",
    )

    torchinfo.summary(model, input_size=(1, 1, 8, 8))  # (B, C, H, W)

    tb_logger = pl_loggers.TensorBoardLogger(args.save_dir, name=experiment_name)

    # Stop training when model converges
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")

    # Save top-1 checkpoints based on Val/Loss
    top1_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(args.save_dir, experiment_name),
        filename='model_best',
        auto_insert_metric_name=False,
    )
    print(f'Saving to dir: {os.path.join(args.save_dir, experiment_name)}')

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=tb_logger,
        callbacks=[top1_checkpoint_callback, early_stop_callback],
    )

    # ------------------------
    # 3 TRAIN MODEL
    # ------------------------
    if args.train:
        trainer.fit(model=model, datamodule=data_module)

    # ------------------------
    # 4 EVALUTE MODEL
    # ------------------------
    if args.train or args.evaluate:
        if args.experiment_name != "":
            experiment_name = args.experiment_name
            checkpoint_file = os.path.join(args.save_dir, experiment_name, 'model_best.ckpt')
            print('Loading checkpoint...', checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])
        # Need val_sum to compute EMD
        _, val_sum = data_module.get_val_max_and_sum()
        model.set_val_sum(val_sum)
        data_module.setup("test")
        test_results = test_model(model, data_module.test_dataloader())
        
        elapsed_time = time.time() - start_time

        with open(os.path.join(args.save_dir, "summary.txt"), "a") as f:
            f.write("=====================================================\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"\tTest score: {str(test_results)}\n")
            f.write(f"\tQuantize: {args.quantize}\n")
            if args.quantize:
                f.write(f"\tBitwidth: {args.bitwidth}\n")
            f.write(f"\tExecution Time (s): {elapsed_time}\n")
            f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ["HAWQ_JET_TAGGING"], "checkpoints/econ"))
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument('--bitwidth', metavar='value', type=int, nargs='+', default=[], help='quantize model to N-bit fixed point')
    parser.add_argument("--quantize", action="store_true", default=False, help="quantize model to N-bit fixed point")

    # Add dataset-specific args
    parser = AutoEncoderDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    if len(args.bitwidth) == 2:
        args.quantize = True
    elif len(args.bitwidth) != 0:
        print(f"Error: Incorrect number of values {len(args.bitwidth)} passed.")
        print("\tOption --bitwidth must be given integer values.")
        sys.exit(1)

    main(args)


"""
    Train FP32: python train.py --train 
    Train INT:  python train.py --train --bitwidth 4, 5
    Eval FP32:  python train.py --evaluate --experiment_name 10.31.2023-18.50.41
    Eval INT:   python train.py --evaluate --experiment_name 10.31.2023-18.50.41 --bitwidth 4, 5
"""
