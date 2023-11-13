"""
 Compute the Hessian. 
"""

########################################################################
# import python standard libary
########################################################################
import os 
import argparse
import logging
# Configure the logging module
logging.basicConfig(filename='hessian.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
########################################################################


########################################################################
# import additional libraries 
########################################################################
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#DB4437", "#4285F4", "#F4B400", "#0F9D58", "purple", "goldenrod", "peru", "coral","turquoise",'gray','navy','m','darkgreen','fuchsia','steelblue'])
from pyhessian import hessian
########################################################################


########################################################################
# import models and dataloaders 
########################################################################
from model import AutoEncoder
from autoencoder_datamodule import AutoEncoderDataModule
########################################################################


########################################################################
# Helper functions 
########################################################################
def dataset_split(dataset, val_split=0.1):
    length = int(len(dataset) * val_split)
    val_dataset = dataset[-length:]
    train_dataset = dataset[:-length]
    return train_dataset, val_dataset


def log_args(args):
    logging.info("########################################################################")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
        print(arg, getattr(args, arg))


def load_dataset(args):
    data_module = AutoEncoderDataModule.from_argparse_args(args)
    data_module.train_bs = args.mini_hessian_batch_size 
    data_module.test_bs = args.mini_hessian_batch_size
    data_module.setup(stage=None)
    dataloader = data_module.val_dataloader()
    return dataloader


def load_model():
    model = AutoEncoder(
        accelerator="auto", 
        quantize=False,
        precision=[32, 32, 32],
        learning_rate=1e-3,  
        econ_type="baseline",
    )
    return model


def get_criterion():
    return nn.MSELoss()
########################################################################


########################################################################
# main
########################################################################
def main(args):
    #######################################
    # Load dataset 
    #######################################
    train_loader = load_dataset(args=args)

    assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
    assert (50000 % args.hessian_batch_size == 0) 
    batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

    if batch_num == 1:
        for inputs in train_loader:
            hessian_dataloader = (inputs, inputs)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(train_loader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break

    #######################################
    # Load model & criterion
    #######################################
    model = load_model()
    print(model)
    criterion = get_criterion()

    if args.resume == '':
        raise Exception("please choose the trained model")

    try:
        model.load_state_dict(torch.load(args.resume))
    except Exception as e:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    #######################################
    # Begin the computation
    #######################################
    # turn model to eval mode
    model.eval()
    if batch_num == 1:
        hessian_comp = hessian(model,
                            criterion, 
                            data=hessian_dataloader,
                            cuda=args.cuda)
    else:
        hessian_comp = hessian(model,
                            criterion,
                            dataloader=hessian_dataloader,
                            cuda=args.cuda)

    print('********** finish data londing and begin Hessian computation **********')

    top_eigenvalues, top_eigenvector, eigenvalueL, eigenvectorL = hessian_comp.eigenvalues()
    trace, traceL = hessian_comp.trace()

    #######################################
    # Save results 
    #######################################
    plt.figure(figsize=(12,12))
    traces = [np.mean(trace_vhv) for trace_vhv in traceL.values()]
    plt.plot(traces, 'o-')
    plt.xlabel('Layers')
    plt.ylabel('Average Hessian Trace')
    plt.xticks(list(range(len(traces))), hessian_comp.layers)
    plt.grid()
    plt.savefig(os.environ["HAWQ_JET_TAGGING"]+'/results/hessian/econ_trace.png')
    plt.yscale('log')
    plt.savefig(os.environ["HAWQ_JET_TAGGING"]+'/results/hessian/econ_trace-log.png')
    plt.close()

    logging.info(f'***Top Eigenvalues:{top_eigenvalues}')
    logging.info(f'***Trace: {np.mean(trace)}')
    logging.info(f'Traces: {traces}')
    logging.info(f'Layers: {hessian_comp.layers}')
    logging.info('########################################################################\n\n')


########################################################################
# arguments 
########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Hessian')
    parser.add_argument('--mini-hessian-batch-size',
                        type=int,
                        default=200,
                        help='input batch size for mini-hessian batch (default: 200)')
    parser.add_argument('--hessian-batch-size',
                        type=int,
                        default=200,
                        help='input batch size for hessian (default: 200)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-norm',
                        action='store_false',
                        help='do we need batch norm or not')
    parser.add_argument('--residual',
                        action='store_false',
                        help='do we need residual connect or not')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='do we use gpu or not')
    parser.add_argument('--resume',
                        type=str,
                        default='',
                        help='get the checkpoint')
    parser.add_argument('--data',
                        type=str,
                        default='',
                        help='path to dataset')
    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='config for selecting dataset features')

    parser = AutoEncoderDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    log_args(args)

    # set random seed to reproduce the work
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    main(args)

# python hessian.py --model econ --resume /data/jcampos/hawq-jet-tagging/checkpoints/econ/10.31.2023-18.50.41/last.ckpt
