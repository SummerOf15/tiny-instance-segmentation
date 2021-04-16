""" Creates a logging utilities such as check point saver and tensorboard summary writer.

Specifically defined for convenience of p_detector.
"""

import torch
import os

from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile


class SaveAndLoad(object):
    """
    CheckLogger saves and loads the models checkpoints.
    We save (i) best state, (ii) latest state of network, 
    model and optimizer state dict, and additional info such as nr. epochs.
    """
    def __init__(self, checkpth):
        self.state = None
        self.path = checkpth
        
        # create directory if not existing
        if not os.path.isdir(checkpth):
            os.mkdir(checkpth)
            
    def state_update(self, epoch, valid_loss, model, optimizer):
        self.state = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
            
    def save_ckp(self, is_best):
        """
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model
        """
        f_path = self.path + "/checkpoint.pt"
        # save checkpoint data to the path given, checkpoint_path
        torch.save(self.state, f_path)
        # if it is a best model, min validation loss
        if is_best:
            best_path = self.path + "/minval"
            # create directory for the best model if not existing
            if not os.path.isdir(best_path):
                os.mkdir(best_path)
                print("Created Directory")
            # copy that checkpoint file to best path given, best_model_path
            copyfile(f_path, best_path + "/minloss_checkpoint.pt")
            
    def load_ckp(self, is_best):
        """
        is_best: load the best model for inference
        model: model that we want to load checkpoint parameters into       
        optimizer: optimizer we defined in previous training
        """
        # set the file path
        if is_best:
            f_path = self.path + "/minval" + "/minloss_checkpoint.pt"
        f_path = self.path + "/checkpoint.pt"
        # load check point
        checkpoint = torch.load(f_path)
        
        return checkpoint


class BoardLogger(object):
    """ a basic wrapper class for customization of tensorboard.
    1. we log train and validation loss.
    2. we log figures for inspection
    """
    def __init__(self, checkpth):
        self.path = checkpth
        self.writer = SummaryWriter(self.path + "/summary")
        
        # create directory if not existing
        if not os.path.isdir(checkpth):
            os.mkdir(checkpth)
            
    def lg_scalar(self, s_name, s_value, n_iter):
        self.writer.add_scalar(s_name, s_value, n_iter)
        
    def lg_figure(self, f_name, f_value):
        """
        f_name: string - identifier
        f_value: figure - matplotlib object
        """
        self.writer.add_figure(f_name, f_value)
        
    def lg_graph(self, model):
        self.writer.add_graph(model)
