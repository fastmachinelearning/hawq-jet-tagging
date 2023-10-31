import os
import argparse
import sys 
sys.path.append("..")

import torch
import torch.nn as nn

from hawq.utils.export import ExportManager

from models.three_layer import get_model
from models.q_three_layer import QThreeLayer_BN, QThreeLayer, QThreeLayer_BNFold
from utils.train_utils import config_model, validate, load_dataset

from training.jet_tagging.config import *


parser = argparse.ArgumentParser(description="Load Checkpoints")
parser.add_argument("--eval", action="store_true", help="Evaluate model.")
parser.add_argument(
    "--save-path",
    type=str,
    default="checkpoints/",
    help="Path to save the quantized model.",
)
parser.add_argument(
    "--batch-norm",
    action="store_true",
    help="Implement with batch normalization.",
)
# args.batch_norm_fold
parser.add_argument(
    "--batch-norm-fold",
    action="store_true",
    help="Implement with batch normalization.",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=50,
    type=int,
    metavar="N",
    help="Print frequency (default: 10)",
)
args = parser.parse_args()


config_map = {
    "w4a6": "../config/config_w4a6.yml",
    "w5a7": "../config/config_w5a7.yml",
    "w6a8": "../config/config_w6a8.yml",
    "w7a9": "../config/config_w7a9.yml",
    "w8a10": "../config/config_w8a10.yml",
    "w4a7": "../config/config_w4a7.yml",
    "w5a8": "../config/config_w5a8.yml",
    "w6a9": "../config/config_w6a9.yml",
    "w7a10": "../config/config_w7a10.yml",
    "w8a11": "../config/config_w8a11.yml",
    "test": "../config/config_mixed.yml",
}


def get_quantized_model(args, model=None):
    if model == None:
        model = get_model(args)
    if args.batch_norm_fold:
        return QThreeLayer_BNFold(model)
    if args.batch_norm:
        return QThreeLayer_BN(model)
    return QThreeLayer(model)


def update_fc_scaling(model, state_dict):
    # fc_scaling_factor saved as scalar tensor, model expects array
    state_dict['dense_1.fc_scaling_factor'] = torch.ones(model.dense_1.fc_scaling_factor.shape)*state_dict['dense_1.fc_scaling_factor']
    state_dict['dense_2.fc_scaling_factor'] = torch.ones(model.dense_2.fc_scaling_factor.shape)*state_dict['dense_2.fc_scaling_factor']
    state_dict['dense_3.fc_scaling_factor'] = torch.ones(model.dense_3.fc_scaling_factor.shape)*state_dict['dense_3.fc_scaling_factor']
    state_dict['dense_4.fc_scaling_factor'] = torch.ones(model.dense_4.fc_scaling_factor.shape)*state_dict['dense_4.fc_scaling_factor']


def main():

    model = get_quantized_model(args)

    # obtain list of all directories (bit configurations) 
    config_dirs = os.listdir(checkpoint_dir)

    # find all non directories 
    for dir in config_dirs:
        if os.path.isdir(os.path.join(checkpoint_dir, dir)) is False:
            print(f"Found file: {dir}")

    # find configuration
    # set precision
    config_model(model, config_dict['w4a7_w4a7_w4a7_w4a7'])

    # load weights 
    try:
        checkpoint = torch.load(args.save_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']

        update_fc_scaling(model, state_dict)
        model.load_state_dict(state_dict)
        model.eval()
    except:
        raise Exception(f'Error encountered loading {args.save_path}')

    # evaluate checkpoint 
    if args.eval:
        criterion = nn.BCELoss()
        _, val_loader = load_dataset("../dataset", 1024, config_map['w4a6'])
        acc = validate(val_loader, model, criterion, args)
        print(f"Checkpoint Accuracy: {acc}")

    print(model.quant_input.act_scaling_factor)
    manager = ExportManager(model)
    manager.export(
       torch.randn([1, 16]),  # input for tracing
       args.save_path.replace('model_best.pth.tar', 'hawq2qonnx_model_v2.onnx'),
    )


if __name__ == "__main__":
    if os.path.isfile(args.save_path):
        main()
    else:
        raise Exception(f"{args.save_path} is not a file.")
