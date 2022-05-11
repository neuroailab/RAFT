#!/bin/bash
mkdir -p checkpoints
# 5.11.22 FOR HONGLIN
python -u train_static_to_motion.py --name centroids-rnd0-movi_d-bs2-small-noEisen-noBootstrap-maskTarget-0 --stage movi_d --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --motion_mask_target --predict_mask
