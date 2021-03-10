#!/usr/bin/env bash
# arch ['mgn', 'dualmgn', 'res50']
#DATASET ['market','cuhk03','dukemtmc','veri','vehicleid']
python main.py \
    --mode train \
    --data_path ~/research_dataset/person_dataset \
    --dataset_name market \
    --height 384 \
    --width 384 \
    --arch dualmgnv \
    --batch_size  8 \
    --num_instances 4 \
    --epoch 1  \
    --test_interval 1 \
    --gpu_devices 0,1



