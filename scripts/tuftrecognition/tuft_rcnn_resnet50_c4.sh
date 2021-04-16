#!/bin/sh

source /volume/USERSTORE/lee_jn/miniconda3/bin/activate p_detector

export CUDA_HOME=/

DIR=/volume/USERSTORE/lee_jn/p_detector

TORCH_HOME=$DIR ; python $DIR/tools/tuft_detection.py --num-gpus 1 --dist-url tcp://127.0.0.1:52112 --config-file $DIR/configs/haicu/faster_rcnn_resnet50_c4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 OUTPUT_DIR /home_local/lee_jn/experiments/haicu/res50x4 DATA_DIR /home_local/lee_jn/data_tuft