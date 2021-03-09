#!/usr/bin/env bash
# arch ['mgn', 'dualmgn', 'res50']
python main.py \
    --mode train \
    --data_path ~/research_dataset/person_dataset/Market \
    --dataset_name Market \
    --height 384 \
    --width 128 \
    --arch mgn \
    --batchid  5 \
    --batchimage 6 \
    --epoch 2  \
    --test_interval 1 \
    --gpu_devices 0,1



