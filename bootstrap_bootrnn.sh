#!/bin/bash
mkdir -p checkpoints
# 5.11.22
python -u train_bootrnn.py --name topobootrnn-movi_d-bs4-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students
