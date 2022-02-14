#!/bin/bash
mkdir -p checkpoints
# 2.14.22
python -u train.py --name thingness-tdw-selfsup-bs4-small-20frames-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 200000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --small --max_frame 25 --full_playroom --filepattern *[0-8]
# 2.13.22
# python -u train.py --name thingness-tdw-selfsup-bs4-small-20frames-fullplay --stage tdw --validation z --gpus 0 1 2 3 --num_steps 200000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --small --max_frame 25 --full_playroom
# 2.11.22
# python -u train.py --name thingness-tdw-selfsup-bs2-small-pw5 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --small --pos_weight 5.0
# python -u train.py --name thingness-tdw-selfsup-bs2-small-20frames --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --small --max_frame 25
# python -u train.py --name thingness-tdw-selfsup-bs2-pw10 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --pos_weight 10.0
# python -u train.py --name thingness-tdw-selfsup-bs2-small --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --small
# python -u train.py --name thingness-tdw-selfsup-bs4-20frames --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --max_frame 25
# 2.10.22
# python -u train.py --name thingness-tdw-selfsup-bs4-cont --restore_ckpt checkpoints/45000_thingness-tdw-selfsup-bs4.pth --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18
# python -u train.py --name thingness-roboall-selfsup-bs4 --stage robonet --validation z --gpus 0 1 2 3 --num_steps 20000 --batch_size 4 --lr 0.0004 --image_size 240 320 --wdecay 0.0001 --model thingness --teacher_ckpt models/raft-sintel.pth --teacher_iters 18
# python -u train.py --name thingness-roboberk-selfsup-bs4 --stage robonet --validation z --gpus 0 1 2 3 --num_steps 5000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt models/raft-sintel.pth --teacher_iters 18 --dataset_names berk*sawy*
# python -u train.py --name thingness-tdw-selfsup-bs4 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18
# 2.9.22
# python -u train.py --name thingness-tdw-continue2 --restore_ckpt checkpoints/thingness-tdw-continue.pth --stage tdw --validation z --gpus 0 1 --num_steps 75000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness
