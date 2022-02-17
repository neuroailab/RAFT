#!/bin/bash
mkdir -p checkpoints
# 2.9.22
python -u train.py --name raft-davis-small-bs1 --stage davis --validation z --gpus 0 --num_steps 100 --batch_size 1 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --small
