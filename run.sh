#!/bin/bash 

allConfigs=("w4a6" "w4a7" "w5a7" "w5a8" "w6a8" "w6a9" "w7a9" "w7a10" "w8a10" "w8a11")


# BN+L1
for i in ${!allConfigs[@]}; do
  echo "running $i - ${allConfigs[$i]}"
  python train_jettagger.py \
    --epochs 100 \
    --lr 0.001 \
    --data /data1/jcampos/datasets \
    --batch-size 1024 \
    --save-path brute_force/${allConfigs[$i]}  \
    --config config/config_${allConfigs[$i]}.yml \
    --batch-norm \
    --l1 \
    --fp-model checkpoints/float/09212022_120558/model_best.pth.tar \
    --roc-precision ${allConfigs[$i]} \
    --print-freq 50
done


# BN
for i in ${!allConfigs[@]}; do
  echo "running $i - ${allConfigs[$i]}"
  python train_jettagger.py \
    --epochs 100 \
    --lr 0.001 \
    --data /data1/jcampos/datasets \
    --batch-size 1024 \
    --save-path brute_force/${allConfigs[$i]}  \
    --config config/config_${allConfigs[$i]}.yml \
    --batch-norm \
    --fp-model checkpoints/float/09212022_120558/model_best.pth.tar \
    --roc-precision ${allConfigs[$i]} \
    --print-freq 50
done


# L1
for i in ${!allConfigs[@]}; do
  echo "running $i - ${allConfigs[$i]}"
  python train_jettagger.py \
    --epochs 100 \
    --lr 0.001 \
    --data /data1/jcampos/datasets \
    --batch-size 1024 \
    --save-path brute_force/${allConfigs[$i]}  \
    --config config/config_${allConfigs[$i]}.yml \
    --l1 \
    --fp-model checkpoints/float/09212022_124743/model_best.pth.tar \
    --roc-precision ${allConfigs[$i]} \ 
    --print-freq 50
done

