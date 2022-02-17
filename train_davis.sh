#!/bin/bash
mkdir -p checkpoints
# 2.17.22
# python -u train.py --name centroid-davis-selfsup-ccs-bs4 --stage davis --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --scale_centroids --model centroid
python -u train.py --name thingness-davis-selfsup-ccs-bs4 --stage davis --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --model thingness
# python -u train.py --name thingness-davis-selfsup-small-ccs-bs4 --stage davis --validation z --gpus 0 1 2 3 --num_steps 100000 --batch_size 4 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --small --model thingness
# python -u train.py --name thingness-davis-selfsup-small-bs4 --stage davis --validation z --gpus 0 1 2 3 --num_steps 10000 --batch_size 4 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --small --model thingness
# python -u train.py --name centroid-davis-selfsup-sc-bs4 --stage davis --validation z --gpus 0 1 2 3 --num_steps 10000 --batch_size 4 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --scale_centroids --model centroid
# python -u train.py --name centroid-davis-selfsup-bs4 --stage davis --validation z --gpus 0 1 2 3 --num_steps 10000 --batch_size 4 --lr 0.0004 --image_size 270 480 --wdecay 0.0001 --model centroid
