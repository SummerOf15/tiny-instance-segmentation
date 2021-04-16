#!/bin/bash

N_THREADS=4
MEM=10G
N_GPUS=volta:1 # can also include type, e.g. "maxwell:1"
TIME=24:00:00 # this would be 6h, acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
OUTPUT=/home/$USER/start_batch_job-%j.out
PARTITION=RMC-C01-BATCH

sbatch --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=$N_THREADS --mem=$MEM --gres=gpu:$N_GPUS --time=$TIME --output=$OUTPUT ./tuftrecognition/tuft_rcnn_x101_fpn.sh
