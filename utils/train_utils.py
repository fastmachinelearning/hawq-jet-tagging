import os
import time
import shutil
import logging
from datetime import datetime
import json
import yaml

import torch
import torch.optim
import torch.utils.data

from sklearn.metrics import accuracy_score

from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear
from hawq.utils.quantization_utils.quant_modules import freeze_model, unfreeze_model

from models.three_layer import three_layer_mlp
from models.q_three_layer import q_three_layer
from models.q_three_layer_bn import q_three_layer_bn

from .meters import AverageMeter, ProgressMeter
from .jet_dataset import JetTaggingDataset


# ------------------------------------------------------------
# logging
# ------------------------------------------------------------
def reset_logging():
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)


def setup_logger(args):
    # setup logging and checkpoint dirs
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S")

    args.save_path = os.path.join(args.save_path, date_time) + "/"

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    reset_logging()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        filename=args.save_path + "training.log",
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(args)


def log_training(data, filename, save_path):
    try:
        path_to_file = os.path.join(save_path, filename)
        with open(path_to_file, "a") as fp:
            if type(data) == str:
                fp.write(data)
            else:
                json.dump(data, fp)
    except:
        logging.error(f"Error while logging to file {path_to_file}")


def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename + "checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(
            filename + "checkpoint.pth.tar", filename + "model_best.pth.tar"
        )


# ------------------------------------------------------------
# dataset
# ------------------------------------------------------------
def load_data(path, batch_size, data_percentage, config, shuffle=True):
    dataset = JetTaggingDataset(path, config["features"], config["preprocess"])
    dataset_length = int(len(dataset) * data_percentage)
    partial_dataset, _ = torch.utils.data.random_split(
        dataset, [dataset_length, len(dataset) - dataset_length]
    )

    data_loader = torch.utils.data.DataLoader(
        partial_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    return data_loader


def load_dataset(data_path, batch_size, config):
    data_config = open_config(config)["data"]
    train_loader = load_data(
        path=os.path.join(data_path, "train"),
        batch_size=batch_size,
        data_percentage=0.75,
        config=data_config,
    )
    val_loader = load_data(
        path=os.path.join(data_path, "val"),
        batch_size=batch_size,
        data_percentage=1,
        shuffle=False,
        config=data_config,
    )
    return train_loader, val_loader


# ------------------------------------------------------------
# training
# ------------------------------------------------------------
def l1_regualization(model):
    l1_loss = 0
    for name, data in model.named_buffers():
        if "integer" in name:
            l1_loss += torch.abs(data).sum()
    return l1_loss


def l2_regualization(model):
    l2_loss = 0
    for name, data in model.named_buffers():
        if "integer" in name:
            l2_loss += (data**2).sum()
    return l2_loss


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    accuracy = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accuracy],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for i, (X_train, y_train) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(X_train.float())

        loss = criterion(output, y_train.float())
        if args.l1:
            loss += 0.000001 * l1_regualization(model)
        if args.l2:
            loss += 0.01 * l2_regualization(model)

        # measure accuracy and record loss
        losses.update(loss.item(), X_train.size(0))

        batch_preds = torch.max(output, 1)[1]
        batch_labels = torch.max(y_train, 1)[1]
        batch_acc = accuracy_score(
            batch_labels.detach().numpy(), batch_preds.detach().numpy()
        )
        # update progress meter
        accuracy.update(batch_acc, X_train.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    scheduler.step()
    return losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    accuracy = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, accuracy], prefix="Test: "
    )

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X_val, y_val) in enumerate(val_loader):
            # compute output
            output = model(X_val.float())

            loss = criterion(output, y_val.float())

            # measure accuracy and record loss
            losses.update(loss.item(), X_val.size(0))

            batch_preds = torch.max(output, 1)[1]
            batch_labels = torch.max(y_val, 1)[1]
            batch_acc = accuracy_score(
                batch_labels.detach().numpy(), batch_preds.detach().numpy()
            )
            # update progress meter
            accuracy.update(batch_acc, X_val.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        logging.info(" * Acc {accuracy.avg:.3f}".format(accuracy=accuracy))

    torch.save(
        {
            "convbn_scaling_factor": {
                k: v
                for k, v in model.state_dict().items()
                if "convbn_scaling_factor" in k
            },
            "fc_scaling_factor": {
                k: v for k, v in model.state_dict().items() if "fc_scaling_factor" in k
            },
            "weight_integer": {
                k: v for k, v in model.state_dict().items() if "weight_integer" in k
            },
            "bias_integer": {
                k: v for k, v in model.state_dict().items() if "bias_integer" in k
            },
            "act_scaling_factor": {
                k: v for k, v in model.state_dict().items() if "act_scaling_factor" in k
            },
        },
        args.save_path + "quantized_checkpoint.pth.tar",
    )

    unfreeze_model(model)
    return accuracy.avg


# ------------------------------------------------------------
# bit configuration
# ------------------------------------------------------------
def open_config(filename):
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def config_quant_act(quant_layer, bit_config):
    quant_layer.activation_bit = bit_config["activation_bit"]


def config_quant_linear(quant_layer, bit_config):
    quant_layer.weight_bit = bit_config["weight_bit"]
    quant_layer.bias_bit = bit_config["bias_bit"]


def config_model(model, filename):
    bit_config = open_config(filename)["model"]
    for layer_name in bit_config.keys():
        quant_layer = getattr(model, layer_name)
        quant_layer_type = type(quant_layer)
        if quant_layer_type == QuantLinear:
            config_quant_linear(quant_layer, bit_config[layer_name])
        elif quant_layer_type == QuantAct:
            config_quant_act(quant_layer, bit_config[layer_name])
        else:
            print(f"Unrecognized layer type {layer_name} - {quant_layer_type}")


# ------------------------------------------------------------
# load and config model
# ------------------------------------------------------------
def load_model(args):
    config = open_config(args.config)["model"]
    if config:
        if args.batch_norm:
            model = q_three_layer_bn()
        else:
            model = q_three_layer()
        config_model(model, args.config)
        return model
    else:
        return three_layer_mlp()
