from __future__ import print_function
import argparse
import torch
import torchvision
import os
import sys

from p_detector.network import fast_rcnn_resnet50
from p_detector.dataset import ArchesDetection, PennFudanDataset
from p_detector.logger import SaveAndLoad, BoardLogger
from p_detector.utils import get_transform
from p_detector.engine import train_one_epoch, evaluate

import p_detector.utils as utils


# define argparse variables
parser = argparse.ArgumentParser(description='Train Fast-RCNN on rmc-slurm - arches demo mission')
parser.add_argument('--data-dir', type=str, default="/home_local/lee_jn/p_detector/notebooks/ardea/set1", metavar='N',
                    help='data directory with both train and validation')
parser.add_argument('--ckp-dir', type=str, default="/home_local/lee_jn/p_detector/logs/ardea", metavar='N',
                    help='data directory with both train and validation')
parser.add_argument('--batchsize', type=int, default=4, metavar='S',
                    help='batch size (default: 1)')
parser.add_argument('--num_epochs', type=int, default=200, metavar='S',
                    help='number of epochs (default: 50)')
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
dataset = ArchesDetection(root=args.data_dir, transforms=get_transform(train=True))
dataset_test = ArchesDetection(root=args.data_dir, transforms=get_transform(train=False))

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

# ckp logger instantiate
ckp_logger = SaveAndLoad(checkpth=args.ckp_dir)
summary = BoardLogger(checkpth=args.ckp_dir)

#### first use old one, and ten try to debug later.
# model definition
model = fast_rcnn_resnet50(progress=True, num_classes=2, pretrained_backbone=True)
# copy the model to gpu
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# training/epochs
for epoch in range(args.num_epochs):
    # train for one epoch, printing every 10 iterations
    metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluator = evaluate(model, data_loader_test, device=device)
    # logging - valid_loss we use train_loss instead
    ckp_logger.state_update(epoch=epoch, valid_loss=metrics.meters['loss'].value, model=model, optimizer=optimizer)
    if epoch ==  0:
        loss_temp = metrics.meters['loss'].value
    if loss_temp > float(metrics.meters['loss'].value):
        ckp_logger.save_ckp(True)
    ckp_logger.save_ckp(False)
    # update the loss temp
    loss_temp = metrics.meters['loss'].value
    # add the loss to the summary
    summary.lg_scalar("loss", metrics.meters['loss'].value, epoch)
    summary.lg_scalar("loss_classifier", metrics.meters['loss_classifier'].value, epoch)
    summary.lg_scalar("loss_box_reg", metrics.meters['loss_box_reg'].value, epoch)
    summary.lg_scalar("loss_objectness", metrics.meters['loss_objectness'].value, epoch)
    summary.lg_scalar("loss_rpn_box_reg", metrics.meters['loss_rpn_box_reg'].value, epoch)
    summary.lg_scalar("lr", metrics.meters['lr'].value, epoch)