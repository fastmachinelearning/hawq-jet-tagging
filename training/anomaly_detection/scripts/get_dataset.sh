#!/bin/sh

URL1="https://zenodo.org/records/3678171/files/dev_data_ToyCar.zip"
URL2="https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip"
ZIPFILE="dev_data_ToyCar.zip"

wget $URL1 -P "$HAWQ_JET_TAGGING/data/anomaly" 
unzip "$HAWQ_JET_TAGGING/data/anomaly/dev_data_ToyCar" -d "$HAWQ_JET_TAGGING/data/anomaly"
rm "$HAWQ_JET_TAGGING/data/anomaly/dev_data_ToyCar.zip"

wget $URL2 -P "$HAWQ_JET_TAGGING/data/anomaly"
unzip "$HAWQ_JET_TAGGING/data/anomaly/eval_data_train_ToyCar" -d "$HAWQ_JET_TAGGING/data/anomaly"
rm "$HAWQ_JET_TAGGING/data/anomaly/eval_data_train_ToyCar.zip"
