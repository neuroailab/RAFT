#!/bin/bash
mkdir -p checkpoints
# 2.10.22
python -u train.py --name thingness-tdw-selfsup-bs4 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18
# 2.9.22
# python -u train.py --name thingness-tdw-continue2 --restore_ckpt checkpoints/thingness-tdw-continue.pth --stage tdw --validation z --gpus 0 1 --num_steps 75000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness
