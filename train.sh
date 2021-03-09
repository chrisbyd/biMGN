#!/usr/bin/env bash
# arch ['mgn', 'dualmgn', 'res50']
python main.py \
    --mode train \
    --data_path ~/research_dataset/person_dataset/Market \
    --dataset_name Market \
    --height 384 \
    --width 384 \
    --arch dualmgn \
    --batchid  4 \
    --batchimage 2 \
    --epoch 2  \
    --test_interval 1 \
    --gpu_devices 0,1



