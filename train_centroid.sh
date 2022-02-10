#!/bin/bash
mkdir -p checkpoints
# 2.10.22
python -u train.py --name centroid-tdw --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid