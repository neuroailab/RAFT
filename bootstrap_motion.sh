#!/bin/bash
mkdir -p checkpoints
# 5.8.22
python -u train.py --name motion-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-0 --stage movi_d --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --teacher_downsample_factor 1 --motion_thresh 0.5 --boundary_thresh 0.5 --gate_stride 1 --motion_ckpt checkpoints/100000_motion-rnd0-movi_d-bs2-small-dtarg-nthr0-cthr025-pr1-gs1-0.pth --flow_ckpt checkpoints/100000_flowBoundary-rnd1-movi_d-bs2-small-mt05-bt05-flit24-gs1-pretrained-1.pth --restore_ckpt checkpoints/100000_motion-rnd0-movi_d-bs2-small-dtarg-nthr0-cthr025-pr1-gs1-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd0-movi_d-bs2-small-dtarg-nthr0-cthr075-pr1-gs1-3.pth --flow_iters 24 --bootstrap --small
# 5.2.22
# python -u train.py --name motion-rnd1-tdw-bs2-large-mt05-bt05-tds2-splitfourall-pretrained-0 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --teacher_downsample_factor 2 --motion_thresh 0.5 --boundary_thresh 0.5 --motion_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd0-tdw-bs4-small-dtarg-nthr0-cthr075-pr1-tds2-fullplayall.pth --flow_ckpt checkpoints/42500_flowBoundary-rnd1-tdw-bs2-large-tds2-splitfourall-pretrained-0.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500 --bootstrap
# python -u train.py --name motion-rnd1-movi_d-bs2-large-mt05-bt05-flit24-gs1-pretrained-0 --stage movi_d --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --teacher_downsample_factor 1 --motion_thresh 0.5 --boundary_thresh 0.5 --gate_stride 1 --motion_ckpt checkpoints/55000_motion-rnd0-movi_d-bs2-large-dtarg-nthr0-cthr025-pr1-gs1-0.pth --boundary_ckpt checkpoints/100000_boundaryMotionReg-rnd0-movi_d-bs2-small-dtarg-nthr0-cthr075-pr1-gs1-3.pth --flow_ckpt checkpoints/60000_flowBoundary-rnd1-movi_d-bs2-large-mt05-bt01-flit24-gs1-pretrained-0.pth --val_freq 2500 --restore_ckpt checkpoints/55000_motion-rnd0-movi_d-bs2-large-dtarg-nthr0-cthr025-pr1-gs1-0.pth --flow_iters 24 --bootstrap
# 4.18.22
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters1000-tds2-fullplayall-pretrained-lr4 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters2500-tds2-fullplayall-pretrained-lr4 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 2500 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters2500-tds2-fullplayall-pretrained-lr4 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 2500 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters1000-tds2-fullplayall-pretrained --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr075-cthr05-iters1000-tds2-fullplayall-pretrained --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.75 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayall-pretrained1 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayall-pretrained-lr4 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters1000-tds2-fullplayall-pthr300 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500 --pixel_thresh 300
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters1000-tds2-fullplayall-pthr1000 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500 --pixel_thresh 1000
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayall-pthr1000 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500 --pixel_thresh 1000
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayall-pthr300 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500 --pixel_thresh 300
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters1000-tds2-fullplayFr6 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayall-pretrained --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 2500
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall-rerun --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.25 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --val_freq 2500
# 4.17.22
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr05-cthr05-iters1000-tds2-fullplayFr6 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.5 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth
# python -u train.py --name motion-rnd1-tdw-bs2-large-dtarg-nthr09-cthr05-iters1000-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.9 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth
# python -u train.py --name boottest4 --stage tdw --validation z --gpus 0 --num_steps 50000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.1 --target_thresh 0.5 --num_propagation_iters 1000 --num_samples 8 --patch_radius 1 --full_playroom --filepattern 0009 --teacher_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth --val_freq 200
# python -u train.py --name noboottest0 --stage tdw --validation z --gpus 0 1 --num_steps 50000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.25 --num_propagation_iters 200 --num_samples 8 --patch_radius 1 --full_playroom --filepattern 0009 --val_freq 200
# 4.17.22
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall-ctd --stage tdw --validation z --gpus 0 1 --num_steps 75000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.25 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --restore_ckpt checkpoints/25000_motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall.pth
# python -u train.py --name motion-rnd0-tdw-bs4-large-imtarg-thr0-pr1-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.5 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8]
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-iters0-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.25 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8] --num_propagation_iters 0
# 4.16.22
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr05-pr1-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.5 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8]
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-nthr0-cthr025-pr1-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.25 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8]
# 4.14.22
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-cthr05-pr1-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8]
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-cthr075-pr1-tds2-fullplayFr6 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.75 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8]
# 4.13.22
# python -u train.py --name motion-rnd0-tdw-bs4-large-dtarg-cthr075-pr1-tds2-fullplayall --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.75 --num_samples 8 --patch_radius 1 --full_playroom --filepattern *[0-8]
# python -u train.py --name motion-rnd0-tdw-bs4-small-dtarg-cthr075-pr1-tds2 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.75 --num_samples 8 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs1-small-allframes-dtarg-cthr05-pr1 --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --small --diffusion_target --downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 4 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs4-small-allframes-dtarg-cthr05-pr1-tds2 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 0 --small --diffusion_target --teacher_downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 8 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs4-small-dtarg-cthr05-pr1-tds2 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --diffusion_target --downsample_factor 1 --teacher_downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 8 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs1-small-imtarg-thr01-pr1 --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 4 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs1-small-imtarg-thr0-pr1 --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.5 --num_samples 4 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs1-small-imtarg-thr0-pr1-tds2 --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --teacher_downsample_factor 2 --motion_thresh 0.0 --target_thresh 0.5 --num_samples 4 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs1-small-dtarg-cthr05-pr1-1iters --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --diffusion_target --downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 4 --patch_radius 1 --num_propagation_iters 1
# 4.12.22
# python -u train.py --name motion-rnd0-tdw-bs4-small-dtarg-cthr075 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --diffusion_target --downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.75 --num_samples 8
# python -u train.py --name motion-rnd0-tdw-bs1-small-dtarg-cthr075-pr1 --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --diffusion_target --downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.75 --num_samples 4 --patch_radius 1
# python -u train.py --name motion-rnd0-tdw-bs1-small-dtarg-cthr05-pr1 --stage tdw --validation z --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --small --diffusion_target --downsample_factor 2 --motion_thresh 0.01 --target_thresh 0.5 --num_samples 4 --patch_radius 1
# 4.7.22
# python -u train.py --name motion-rnd0-tdw-bs4-small-fullplay-zthr1 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 1.0 --small
# python -u train.py --name motion-rnd0-tdw-bs4-small-fullplay-zthr02 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 0.2 --small
# python -u train.py --name motion-rnd0-tdw-bs4-small-fullplay-zthr3 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 3.0 --small
# python -u train.py --name motion-rnd0-tdw-bs4-small-zthr3 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --motion_thresh 3.0 --small
# 4.6.22
# python -u train.py --name motion-rnd0-tdw-bs4-small-fullplay-th05 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 0.05 --small
# python -u train.py --name occlusion-rnd0-tdw-bs4-small-fullplay-th05 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model occlusion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 0.05 --small
# 4.4.22
# python -u train.py --name motion-sup-tdw-bs8-small-split4 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --supervised --small
# 4.2.22
# python -u train.py --name motion-sup-tdw-bs8-large-split4 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --max_frame 6 --supervised
# python -u train.py --name motion-sup-tdw-bs8-large-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --supervised
# python -u train.py --name motion-sup-tdw-bs8-small-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --supervised --small
# 3.16.22
# python -u train.py --name motion-rnd0-tdw-bs8-large-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 0.01
# 3.15.22
# python -u train.py --name motion-rnd0c-tdw-bs8-small-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 65000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --small --no_aug --full_playroom --filepattern *[0-8] --max_frame 6 --motion_thresh 0.01 --restore_ckpt checkpoints/35000_motion-rnd0-tdw-bs8-small-fullplay-tr0-8.pth
# 3.10.22
# python -u train.py --name motion-molo100-tdw-bs8-small-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --small --no_aug --full_playroom --filepattern *[0-8] --use_motion_loss --loss_scale 100.0
# python -u train.py --name motion-rnd1-tdw-bs2-small-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --small --no_aug --full_playroom --filepattern *[0-8] --motion_thresh 0.1 --motion_ckpt checkpoints/20000_motion-tdw-bs8-small-fullplay-tr0-8.pth --features_ckpt checkpoints/20000_mocentroid-tdw-bs8-fullplay-tr0-8.pth --teacher_iters 1
# python -u train.py --name motion-tdw-bs8-small-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model motion --small --no_aug --full_playroom --filepattern *[0-8]
# 2.27.22
# python -u train.py --name thingness-tdw-selfsupSintel-bs4-small-20frames-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt models/raft-sintel.pth --teacher_iters 18 --small --max_frame 25 --full_playroom --filepattern *[0-8] --restore_ckpt checkpoints/thingness-tdw-selfsupSintel-bs8-small-20frames-fullplay-tr0-8
# 2.14.22
# python -u train.py --name thingness-tdw-selfsupSintel-bs8-small-20frames-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt models/raft-sintel.pth --teacher_iters 18 --small --max_frame 25 --full_playroom --filepattern *[0-8]
# python -u train.py --name thingness-tdw-selfsup-bs4-small-20frames-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 200000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model thingness --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --small --max_frame 25 --full_playroom --filepattern *[0-8]
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
