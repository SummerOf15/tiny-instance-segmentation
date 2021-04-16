#!/bin/sh

source /volume/USERSTORE/lee_jn/miniconda3/bin/activate p_detector

export CUDA_HOME=/

DIR=/volume/USERSTORE/lee_jn/p_detector

TORCH_HOME=$DIR ; python $DIR/tools/sam_detection.py --num-gpus 1 --is-detr True --dist-url tcp://127.0.0.1:52111 --config-file $DIR/configs/sam/detr_resnet50_256_6_6_torchvision.yaml SOLVER.IMS_PER_BATCH 4 OUTPUT_DIR /home_local/lee_jn/experiments/sam/sam_detr_resnet50_aug DATA_DIR /home_local/lee_jn/data_sam trainaugment True
