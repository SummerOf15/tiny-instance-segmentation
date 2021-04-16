#!/bin/sh

source /volume/USERSTORE/lee_jn/miniconda3/bin/activate p_detector

export CUDA_HOME=/

DIR=/volume/USERSTORE/lee_jn/p_detector

TORCH_HOME=$DIR ; python $DIR/tools/ardea_detection.py --num-gpus 1 --dist-url tcp://127.0.0.1:52111 --config-file $DIR/configs/sam/faster_rcnn_resnet50_c4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 OUTPUT_DIR /home_local/lee_jn/experiments/sam/sam_rcnn_resnet50_aug DATA_DIR /home_local/lee_jn/data_sam trainaugment True
