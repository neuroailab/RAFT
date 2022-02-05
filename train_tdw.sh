#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-tdw-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 3
# python -u train.py --name raft-tdw-noaug --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001
# python -u train.py --name raft-tdw-long --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001
