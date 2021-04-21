#!/bin/sh

source /home/$USER/miniconda3/bin/activate tor

echo "=====Job Infos ===="
# echo "Node List: " $SLURM_NODELIST
# echo "jobID: " $SLURM_JOB_ID
# echo "Partition: " $SLURM_JOB_PARTITION
# echo "Submit directory:" $SLURM_SUBMIT_DIR
# echo "Submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: `pwd`"
echo "As the user: `whoami`"
echo "Python version: `python -c 'import sys; print(sys.version)'`"
echo "pip version: `pip --version`"


export CUDA_HOME=/

DIR=/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation

start_time=`date +%s`
echo "Job Started at "`date`

TORCH_HOME=$DIR ; python $DIR/tools/tuft_detection.py --num-gpus 1 --is-detr True --dist-url tcp://127.0.0.1:52111 --config-file $DIR/configs/tuft/detr_resnet50_256_6_6_torchvision.yaml SOLVER.IMS_PER_BATCH 1 OUTPUT_DIR $DIR/experiments/tuft/tuft_detr_resnet50 DATA_DIR $DIR/dataset


echo "Job ended at "`date`
end_time=`date +%s`
total_time=$((end_time-start_time))

echo "Took " $total_time " s"