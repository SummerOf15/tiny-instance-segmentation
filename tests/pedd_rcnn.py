from __future__ import print_function
import argparse
import torch
import torchvision
import os
import sys

from p_detector.dataset import PennFudanDataset
from p_detector.logger import SaveAndLoad, BoardLogger
from p_detector.utils import get_transform
from p_detector.engine import train_one_epoch, evaluate

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import p_detector.utils as utils


# define argparse variables
parser = argparse.ArgumentParser(description='Train Fast-RCNN on rmc-slurm')
parser.add_argument('--data-dir', type=str, default="/home_local/lee_jn/p_detector/notebooks/PennFudanPed", metavar='N',
                    help='data directory with both train and validation')
parser.add_argument('--ckp-dir', type=str, default="/home_local/lee_jn/p_detector/logs/PennRCNN", metavar='N',
                    help='data directory with both train and validation')
parser.add_argument('--batchsize', type=int, default=1, metavar='S',
                    help='batch size (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args, unknown = parser.parse_known_args()
torch.manual_seed(args.seed)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("devices available:", device)


# our dataset has two classes only - background and person
num_classes = 2

# use our dataset and defined transformations
dataset = PennFudanDataset(args.data_dir, get_transform(train=True))
dataset_test = PennFudanDataset(args.data_dir, get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batchsize, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)  
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batchsize, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)


# model definition
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# copy the model to gpu
model.to(device)

# logger definition
# ckp logger instantiate
ckp_logger = SaveAndLoad(checkpth=args.ckp_dir)
board_logger = BoardLogger(checkpth=args.ckp_dir)


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# let's train it for 10 epochs
num_epochs = 5


# training/epochs
# we set it to an arbitrarily high values so that they get initialized.
loss_temp = 100000.0
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluator = evaluate(model, data_loader_test, device=device)
    # logging - valid_loss we use train_loss instead
    ckp_logger.state_update(epoch=epoch, valid_loss=metrics.meters['loss'].value, model=model, optimizer=optimizer)
    # you can also include a if loop here once we get a nice val loss 
    if loss_temp > float(metrics.meters['loss'].value):
        loss_temp = metrics.meters['loss'].value
        ckp_logger.save_ckp(True)
    # otherwise, save the model elsewhere
    ckp_logger.save_ckp(False)
    # if -- summarize (per epoch)
    board_logger.lg_scalar("best_loss", loss_temp, epoch)