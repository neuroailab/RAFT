#!/bin/bash
mkdir -p checkpoints
# 2.9.22
python -u train.py --name thingness-tdw-continue --restore_ckpt checkpoints/5000_thingness-tdw.pth --stage tdw --validation z --gpus 0 1 --num_steps 20000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness
