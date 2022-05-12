#!/bin/bash
mkdir -p checkpoints
# 5.11.22
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_e-bs2-small-noEisen-Bootstrap-maskTarget-pretrained-0 --stage movi_e --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target --bootstrap
python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_e-bs2-small-noEisen-noBootstrap-maskTarget-pretrained-0 --stage movi_e --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target
# 5.10.22 FOR HONGLIN
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs1-small-Eisen-noBootstrap-maskTarget-pretrained-128x128-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target --teacher_downsample_factor 1 --static_ckpt checkpoints/80000_eisen_teacher_v1_128_bs4.pth
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs1-small-Eisen-noBootstrap-noMaskTarget-pretrained-128x128-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --teacher_downsample_factor 1 --static_ckpt checkpoints/80000_eisen_teacher_v1_128_bs4.pth
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs1-small-Eisen-Bootstrap-maskTarget-pretrained-128x128-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --teacher_downsample_factor 1 --static_ckpt checkpoints/80000_eisen_teacher_v1_128_bs4.pth --bootstrap --motion_mask_target
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs1-small-Eisen-Bootstrap-noMaskTarget-pretrained-128x128-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --teacher_downsample_factor 1 --static_ckpt checkpoints/80000_eisen_teacher_v1_128_bs4.pth --bootstrap
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs1-small-Eisen-noBootstrap-maskTarget-pretrained-64x64-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target --teacher_downsample_factor 2 --static_ckpt checkpoints/30000_eisen_teacher_v1_64_bs16.pth
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs1-small-Eisen-Bootstrap-maskTarget-pretrained-64x64-0 --stage movi_d --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target --teacher_downsample_factor 2 --static_ckpt checkpoints/30000_eisen_teacher_v1_64_bs16.pth --bootstrap
# 5.10.22
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs2-small-noEisen-noBootstrap-maskTarget-pretrained-0 --stage movi_d --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target
# python -u train_static_to_motion.py --name flowCentroids-rnd2-movi_d-bs2-small-noEisen-Bootstrap-maskTarget-pretrained-0 --stage movi_d --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model flow_centroids --no_aug --max_frame 0 --teacher_downsample_factor 1 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --val_freq 2500 --small --restore_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --motion_mask_target --bootstrap