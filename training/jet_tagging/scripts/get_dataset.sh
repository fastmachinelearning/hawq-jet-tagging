#!/bin/bash 

SAVE_DIR="$HAWQ_JET_TAGGING/data/jets"

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz -P $SAVE_DIR
tar -zxf "$SAVE_DIR/hls4ml_LHCjet_100p_train.tar.gz" -C $SAVE_DIR
rm "$SAVE_DIR/hls4ml_LHCjet_100p_train.tar.gz"

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz -P $SAVE_DIR
tar -zxf "$SAVE_DIR/hls4ml_LHCjet_100p_val.tar.gz" -C $SAVE_DIR
rm "$SAVE_DIR/hls4ml_LHCjet_100p_val.tar.gz"
