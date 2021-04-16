#!/bin/bash

N_THREADS=32
MEM=85G
N_GPUS=turing:8 # can also include type, e.g. "maxwell:1"
TIME=3-5 #this would be 6h, acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
OUTPUT=/home/$USER/start_batch_job-%j.out
PARTITION=RMC-C01-BATCH

sbatch --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=$N_THREADS --mem=$MEM --gres=gpu:$N_GPUS --time=$TIME --output=$OUTPUT $1
