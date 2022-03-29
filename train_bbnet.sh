#!/bin/bash
mkdir -p checkpoints
# 3.29.22
python -u train_bbnet.py --name bbnet_rnd0static_fullplayFr5_bs4 --model bbnet --train_mode train_static --teacher_model motion --teacher_ckpt checkpoints/motion-rnd0-tdw-bs8-large-fullplay-tr0-8.pth --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5
