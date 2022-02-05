#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-chairs-scratch --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
