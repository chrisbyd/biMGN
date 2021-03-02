#!/usr/bin/env bash
python test.py \
    --src_dataset 'VeRi' \
    --trg_dataset 'VehicleID' \
    --height 384 \
    --width 384 \
    --batch_size 120 \
    --arch 'dualmgn' \
    --resume './out/source_out/dualmgnVeRi_60.pth'\
    --logs_dir ./logs/veri/ \
    --gpu_devices 0 \
    --test_iteration \
    --num_split 1
