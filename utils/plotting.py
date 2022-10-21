import os
import json
import numpy as np
import itertools

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from .train_utils import get_model

# confusion matrix code from Maurizio
# /eos/user/m/mpierini/DeepLearning/ML4FPGA/jupyter/HbbTagger_Conv1D.ipynb
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def plotRoc(fpr, tpr, auc, labels, linestyle, legend=True):
    for i, label in enumerate(labels):
        plt.plot(
            tpr[label],
            fpr[label],
            label="%s tagger, AUC = %.1f%%"
            % (label.replace("j_", ""), auc[label] * 100.0),
            linestyle=linestyle,
        )
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001, 1)
    plt.grid(True)
    if legend:
        plt.legend(loc="upper left")
    plt.figtext(
        0.25,
        0.90,
        "hls4ml",
        fontweight="bold",
        wrap=True,
        horizontalalignment="right",
        fontsize=14,
    )


def rocData(y, predict_test, labels):

    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:, i]
        df[label + "_pred"] = predict_test[:, i]

        fpr[label], tpr[label], threshold = roc_curve(df[label], df[label + "_pred"])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1


def makeRoc(y, predict_test, labels, linestyle="-", legend=True):

    if "j_index" in labels:
        labels.remove("j_index")

    fpr, tpr, auc1 = rocData(y, predict_test, labels)
    plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend)
    return predict_test


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))


def load_training_log(checkpoint_dir):

    input_file = open(checkpoint_dir + "model_acc.json")
    model_acc = json.load(input_file)

    input_file = open(checkpoint_dir + "model_loss.json")
    model_loss = json.load(input_file)
    return model_acc, model_loss


def make_train_plot(args):
    # acc and loss plot
    model_acc, model_loss = load_training_log(args.save_path)

    plt.figure()
    plt.rcParams["figure.figsize"] = [7, 3]
    plt.rcParams["figure.autolayout"] = True

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    (l1,) = ax1.plot(model_acc, "r")
    (l2,) = ax2.plot(model_loss, "c-")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="r")
    ax2.set_ylabel("Loss", color="c")

    plt.legend([l1, l2], ["Accuracy", "Loss"])

    plt.savefig(os.path.join(args.save_path, "training.png"))
    plt.close()


def make_plots(hawq_model, val_loader, args):
    if args.fp_model:
        float_model = get_model(args)
        checkpoint = torch.load(args.fp_model, map_location=torch.device("cpu"))
        float_model.load_state_dict(checkpoint["state_dict"])
        float_model.eval()

    hawq_output = torch.tensor([])
    float_output = torch.tensor([])
    val_output = torch.tensor([])

    with torch.no_grad():
        for X_val, y_val in val_loader:
            # compute output
            hawq_batch = hawq_model(X_val.float())
            if args.fp_model:
                float_batch = float_model(X_val.float())
            # concat
            hawq_output = torch.cat((hawq_output, hawq_batch))
            if args.fp_model:
                float_output = torch.cat((float_output, float_batch))
            val_output = torch.cat((val_output, y_val))

    # make roc curves
    labels = ["g", "q", "w", "z", "t"]
    qlabels = [f"{args.roc_precision} {label}" for label in labels]

    plt.figure()
    fig, ax = plt.subplots(figsize=(9, 9))
    _ = makeRoc(val_output, float_output, labels)
    plt.gca().set_prop_cycle(None)  # reset the colors
    _ = makeRoc(val_output, hawq_output, qlabels, linestyle="--")

    from matplotlib.lines import Line2D

    lines = [Line2D([0], [0], ls="-"), Line2D([0], [0], ls="--")]
    from matplotlib.legend import Legend

    leg = Legend(
        ax,
        lines,
        labels=["32-bit floating point", f"{args.roc_precision} HAWQ"],
        loc="lower right",
        frameon=False,
    )
    ax.add_artist(leg)

    plt.savefig(os.path.join(args.save_path, "roc.png"))
    plt.close()

    # confusion matrix
    cm = confusion_matrix(
        torch.argmax(val_output, dim=1), torch.argmax(hawq_output, dim=1)
    )
    plot_confusion_matrix(cm, labels, normalize=True)
    plt.savefig(os.path.join(args.save_path, "cm.png"))
    plt.close()
