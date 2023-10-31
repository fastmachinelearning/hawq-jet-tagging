import os
import sys 
sys.path.append(os.environ["HAWQ_JET_TAGGING"])
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

from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantBnLinear
from hawq.utils.quantization_utils.quant_modules import freeze_model, unfreeze_model

from model import get_model, get_quantized_model

from utilities.meters import AverageMeter, ProgressMeter
from utilities.jet_dataset import JetTaggingDataset


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
        with open(path_to_file, "w") as fp:
            if type(data) == str:
                fp.write(data)
            else:
                json.dump(data, fp)
    except:
        logging.error(f"Error logging to file {path_to_file}")


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
    dataset, partial_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - dataset_length, dataset_length]
    )

    # return partial_dataset

    if data_percentage < 1:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
        split_loader = torch.utils.data.DataLoader(
            partial_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
        return data_loader, split_loader
    data_loader = torch.utils.data.DataLoader(
        partial_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    return data_loader


def load_dataset(data_path, batch_size, config, train_percentage=1):
    data_config = open_config(config)["data"]
    train_loader = load_data(
        path=os.path.join(data_path, "train"),
        batch_size=batch_size,
        data_percentage=train_percentage,
        shuffle=False,
        config=data_config,
    )
    test_loader = load_data(
        path=os.path.join(data_path, "val"),
        batch_size=batch_size,
        data_percentage=1,
        shuffle=False,
        config=data_config,
    )
    return train_loader, test_loader


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
    accuracy = AverageMeter("Acc", ":6.6f")
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
            loss += args.l1_lambda * l1_regualization(model) 
        if args.l2:
            loss += args.l2_lambda * l2_regualization(model)

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

    if optimizer.param_groups[0]["lr"] > 0.00001:
        scheduler.step()
    return losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    accuracy = AverageMeter("Acc", ":6.6f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, accuracy], prefix="Test: "
    )

    # switch to evaluate mode
    freeze_model(model)
    model.eval()
    correct = 0
    total = 0

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

            total += batch_labels.size(0)
            correct += (batch_preds == batch_labels).sum().item()
            
            # update progress meter
            accuracy.update(batch_acc, X_val.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                print(batch_acc)
        logging.info(" * Acc {accuracy.avg:.6f}".format(accuracy=accuracy))

    unfreeze_model(model)
    print(f"Simplified: {correct/total}")
    print(f"Meter: {accuracy.avg}")
    return correct/total


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


def config_model(model, config):
    if type(config) == str:
        bit_config = open_config(config)["model"]
    elif type(config) == dict:
        bit_config = config
    for layer_name in bit_config.keys():
        quant_layer = getattr(model, layer_name)
        quant_layer_type = type(quant_layer)
        if quant_layer_type == QuantLinear or quant_layer_type == QuantBnLinear:
            config_quant_linear(quant_layer, bit_config[layer_name])
        elif quant_layer_type == QuantAct:
            config_quant_act(quant_layer, bit_config[layer_name])
        else:
            print(f"Unrecognized layer type {layer_name} - {quant_layer_type}")


# ------------------------------------------------------------
# load and config model
# ------------------------------------------------------------
def load_model(args, model=None):
    config = open_config(args.config)["model"]
    if config:
        # model = ThreeLayer_BN()
        # model.load_state_dict(torch.load("/data/jcampos/hawq-jet-tagging/checkpoints/test/11112022_122401/model_best.pth.tar")["state_dict"])
        model = get_quantized_model(args)
        config_model(model, args.config)  # set bitwidth
    else:
        model = get_model(args)
    return model
