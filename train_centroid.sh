#!/bin/bash
mkdir -p checkpoints
# 2.14.22
python -u train.py --name centroid-tdw-selfsup-bs4-scaledmse-fullplay-tr0-8 --stage tdw --validation z --gpus 0 1 2 3 --num_steps 200000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid --scale_centroids --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --full_playroom --filepattern *[0-8]
# 2.13.22
# python -u train.py --name centroid-tdw-selfsup-bs4-scaledmse-fullplay --stage tdw --validation z --gpus 0 1 2 3 --num_steps 200000 --batch_size 4 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid --scale_centroids --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18 --full_playroom
# 2.11.22
# python -u train.py --name centroid-tdw-selfsup-bs2-scaledmse --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 2 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid --scale_centroids --teacher_ckpt checkpoints/raft-tdw-bn-fullplay.pth --teacher_iters 18
# 2.10.22
# python -u train.py --name centroid-tdw-static-unscaled --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid
# python -u train.py --name centroid-tdw-static-scaledmse --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid --scale_centroids
# python -u train.py --name centroid-tdw-scaledabs --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid --scale_centroids
# python -u train.py --name centroid-tdw-scaled --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid --scale_centroids
# python -u train.py --name centroid-tdw-unscaled --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid
# python -u train.py --name centroid-tdw --stage tdw --validation z --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model centroid
