#!/bin/bash
mkdir -p checkpoints
# 6.3.22
python -u train_bootrnn.py --name topobootrnn-0acomp-beta1it20-diff250-movi_d-bs4-rnd2-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students --teacher_config configs/topobootrnn-0acomp-movi_d-rnd1-0.yml --bootstrap
# 6.1.22 # first round of bootstrapping
# python -u train_bootrnn.py --name topobootrnn-0acomp-diff250_movi_d-bs4-rnd0-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students
# python -u train_bootrnn.py --name topobootrnn-0acomp-beta1it20_movi_d-bs4-rnd1-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students --teacher_config configs/topobootrnn-0acomp-movi_d-0.yml --bootstrap
# python -u train_bootrnn.py --name topobootrnn-0acomp-movi_d-bs4-rnd1-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students --teacher_config configs/topobootrnn-0acomp-movi_d-0.yml --bootstrap
# 5.31.22
# take out autocomplete and boundary thresholding
# python -u train_bootrnn.py --name topobootrnn-0acomp-movi_d-bs4-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students
# fixed (hopefully) the orientations target masking
# python -u train_bootrnn.py --name topobootrnn-movi_d-bs4-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2000 --save_students
# 5.30.22
# python -u train_bootrnn.py --name topobootrnn-movi_d-bs2-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootrnn --no_aug --max_frame 0 --val_freq 2500 --save_students
