#!/bin/bash
mkdir -p checkpoints
# 4.1.22
#python -u train_eisen.py --name eisen_unsup --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5
python -u train_eisen.py --name eisen_raft_0.5 --teacher_class raft_pretrained --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5
python -u train_eisen.py --name eisen_unsup_improved --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5

python -u train_eisen.py --name eisen_raft_0.5_bs2 --teacher_class raft_pretrained --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 2 --lr 0.0005 --image_size 512 512 --wdecay 0.000 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5