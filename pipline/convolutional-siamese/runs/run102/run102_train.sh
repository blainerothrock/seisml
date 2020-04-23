#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
num_run=102
run_id="run$num_run"

mkdir -p runs/$run_id
cp train.sh runs/${run_id}/${run_id}_train.sh

#python ../../train.py \
#    --model_type "conv" \
#    --output_directory "runs/$run_id" \
#    --dataset_directory "/home/b/dev/seis/data/triggered_earthquakes/prepared/" \
#    --batch_size 64 \
#    --num_workers 10 \
#    --num_epochs 100 \
#    --transforms "bandpass:whiten" \
#    --augmentations "amplitude:noise" \
#    --length 8192 \
#    --sample_strategy "sequential" \
#    --learning_rate 2e-5 \
#    --weight_decay 1e-1 \
#    --split SAC_20021102_XF_prem \
#    --embedding_size 10 \
#    --loss_function "dpcl"

python ../../test.py \
    --output_directory "runs/$run_id"