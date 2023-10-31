#!/bin/sh

# Download data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P "$HAWQ_JET_TAGGING/data/cifar10"
tar -xvf "$HAWQ_JET_TAGGING/data/cifar-10-python.tar.gz" -C "$HAWQ_JET_TAGGING/data/cifar10"

# Create checkpoints dir 
mkdir "$HAWQ_JET_TAGGING/data/ic"
