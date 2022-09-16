#!/bin/bash 

for ((n=0;n<3;n++))
do 
    echo "Training model with batch normalization..."

    python train_jettagger.py --epochs 100 \
        --lr 0.01 \
        --data /data1/jcampos/datasets \
        --batch-size 1024 \
        --save-path checkpoints/uniform6  \
        --config config/config_6bit.yml \
        --print-freq 50 \
        --batch-norm 
done

