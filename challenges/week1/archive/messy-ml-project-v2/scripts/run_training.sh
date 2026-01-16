#!/bin/bash
# Training script - not sure if this still works

cd ../src
python train_final_REAL.py --epochs 10 --batch-size 64 --lr 0.001

# Old command (don't use):
# python train_final.py
