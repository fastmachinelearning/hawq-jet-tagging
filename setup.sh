#!/bin/sh

######################################################################## 
# SETUP ENVIRONMENT
########################################################################
# Conda 
ENVIRONMENT_NAME="base"
ENVIRONMENT_EXITS=$(conda env list | grep "$ENVIRONMENT_NAME" | wc -l)

echo "CONDA ENVIRONMENT_EXITS: $ENVIRONMENT_EXITS"
if [ "$ENVIRONMENT_EXITS" == "1" ]; then 
    conda activate base
else 
    conda env create -f environment/environment.yml
    conda activate $ENVIRONMENT_NAME
fi

# Variables
export HAWQ_JET_TAGGING=$(pwd)

######################################################################## 
# HELPER FUNCTIONS 
########################################################################

# Usage: get_dataset [ ic | jets | ad | econ ]
get_dataset() {
    if [ "$1" == "jets" ]; then
        echo "jet-tagging"
        DIR="jet_tagging"
    elif [ "$1" == "ic" ]; then
        echo "image-classification"
        DIR="image_classification"
    elif [ "$1" == "ad" ]; then
        echo "anomaly-detection"
        DIR="anomaly_detection"
    elif [ "$1" == "econ" ]; then 
        echo "econ"
        DIR="econ"
    elif [ "help" == "help" ]; then 
        echo "Usage: get_dataset DATASET_NAME" 
        echo "Options:"
        echo -e "\tjets: Classifying simulated jets of proton-proton colisions."
        echo -e "\tecon: Data generated with the Level 1 Trigger Primitives simulation."
        echo -e "\tic  : Image classification on CIFAR10."
        echo -e "\tad  : Unsupervised detection of anomalous sounds."
        return 0
    else
        echo "Error: unrecognized dataset $1"
    fi

    DOWNLOAD_DATA=""$HAWQ_JET_TAGGING/training/$DIR/scripts/get_dataset.sh""
    $DOWNLOAD_DATA
}

# Usage: tensorboard [ ic | jets | ad ] 
tensorboard() { tensorboard --logdir "$HAWQ_JET_TAGGING/checkpoints/$1"; }


# Usage: train [ ic | jets | ad ]
train() {
    if [ "$1" == "jets" ]; then 
        echo "Launching jet tagging scripts."
        # does not work, old scripts have not been moved to new dir 
        TASK=""
    elif [ "$1" == "ic" ]; then
        echo "Launching image classification (ResNet) training scripts."
        cd "$HAWQ_JET_TAGGING/training/image_classification"  # todo: why cd to directory?
        TASK="python train.py --train"
    elif [ "$1" == "ad" ]; then
        echo "Launching anomaly detection (ToyCar only)."
        TASK="python $HAWQ_JET_TAGGING/training/anomaly_detection/train.py"
    fi 
    $TASK 
}

add_permissions()
{
    echo "Adding execute permission to files:"
    for dir in "jet_tagging" "anomaly_detection" "econ" "image_classification"
    do 
        SCRIPT_DIR="$HAWQ_JET_TAGGING/training/$dir/scripts"
        echo -e "\t$SCRIPT_DIR"

        FILES="$SCRIPT_DIR/*"
        for file in $FILES
        do 
            echo -e "\t\t$file"
            chmod +x "$file"
        done 
    done
}

# assume first time running --> add execute permission on scripts 
if [ "$ENVIRONMENT_EXITS" == "0" ]; then 
    add_permissions
fi

######################################################################## 
# SHORTCUTS   
########################################################################
