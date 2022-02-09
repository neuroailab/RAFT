#!/bin/bash
mkdir -p checkpoints
# 2.9.22
python -u train.py --name bootraft-tdw-inpnorm --stage tdw --validation chairs --gpus 0 1 --num_steps 20000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootraft
# 2.8.22
# python -u train.py --name raft-tdw-bn-amaxtotalenergyframes --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --max_frame -1 --training_frames datasets/supervision_frames/model_split_4_totalenergyargmax.json
# python -u train.py --name raft-tdw-bn-amaxflowframes --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --max_frame -1 --training_frames datasets/supervision_frames/model_split_4_threshargmax.json
# 2.7.22
# python -u train.py --name raft-tdw-bn-100clip --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --clip 100.0
# python -u train.py --name raft-tdw-bn-0wd --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0
# python -u train.py --name raft-tdw-bn-allframes --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --max_frame -1
# 2.6.22
# python -u train.py --name bootraft-tdw --stage tdw --validation chairs --gpus 0 1 --num_steps 20000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootraft
# python -u train.py --name bootraft-tdw-noaug --stage tdw --validation chairs --gpus 0 1 --num_steps 20000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --model bootraft --no_aug
# python -u train.py --name raft-tdw-g0-lr0001-2iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --iters 2 --gamma 0
# python -u train.py --name raft-tdw-g0-lr0001-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --iters 3 --gamma 0
# python -u train.py --name raft-tdw-g0-lr0001-4iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --iters 4 --gamma 0
# python -u train.py --name raft-tdw-g045-lr000025-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.000025 --image_size 496 496 --wdecay 0.0001 --iters 3 --gamma 0.45
# 2.5.22
# python -u train.py --name raft-tdw-static --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --static_coords
# python -u train.py --name raft-tdw-static-6iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --static_coords --iters 6
# python -u train.py --name raft-tdw-static-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --static_coords --iters 3
# python -u train.py --name raft-tdw-g045-4iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 4 --gamma 0.45
# python -u train.py --name raft-tdw-g05-6iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 6 --gamma 0.5
# python -u train.py --name raft-tdw-lr0001-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --iters 3
# python -u train.py --name raft-tdw-g045-lr0001-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0001 --image_size 496 496 --wdecay 0.0001 --iters 3 --gamma 0.45
# python -u train.py --name raft-tdw-g045-lr001-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.001 --image_size 496 496 --wdecay 0.0001 --iters 3 --gamma 0.45
# python -u train.py --name raft-tdw-lr001-4iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.001 --image_size 496 496 --wdecay 0.0001 --iters 4
# python -u train.py --name raft-tdw-bn-fullplay --stage tdw --validation chairs --gpus 0 1 --num_steps 200000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --full_playroom
# python -u train.py --name raft-tdw-bn-3iters --stage tdw --validation chairs --gpus 0 1 --num_steps 10000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 3
# python -u train.py --name raft-tdw-bn-4iters --stage tdw --validation chairs --gpus 0 1 --num_steps 10000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 4
# python -u train.py --name raft-tdw-bn-5iters --stage tdw --validation chairs --gpus 0 1 --num_steps 10000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 5
# python -u train.py --name raft-tdw-bn-6iters --stage tdw --validation chairs --gpus 0 1 --num_steps 10000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 6
# python -u train.py --name raft-tdw-bn-8iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 8
# python -u train.py --name raft-tdw-bn-10iters --stage tdw --validation chairs --gpus 0 1 --num_steps 5000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 10
# 2.4.22
# python -u train.py --name raft-tdw-bn --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001
# python -u train.py --name raft-tdw-6iters --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001 --iters 6
# python -u train.py --name raft-tdw-noaug --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001
# python -u train.py --name raft-tdw-long --stage tdw --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 496 496 --wdecay 0.0001
