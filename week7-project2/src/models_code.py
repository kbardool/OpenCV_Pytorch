import os
from pathlib import Path
# from dataclasses import dataclass
# from lightning.pytorch.demos.boring_classes import RandomDataset
# import zipfile
# import torchvision
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.data as data
# import torch.optim

# from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as TVF
# from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
import lightning as L




class PretrainedResNet(nn.Module,ABC):
    """ 
    Abstract base class for pretrained ResNet models with selective fine-tuning.
    """
    def __init__(self, num_classes=13, finetune=None, name = None, weights=None, verbose = False):
        super().__init__()
        # Subclasses must provide these
        model_fn, model_name, weights = self._get_model_config()
        # assert "fc" in finetune, "'fc' must be in fine tuning"
        
        self.name = model_name if name is None else name
        if finetune is not None:
            self.finetune = finetune
            self.tuning_layers = ''.join(
                ['fc'] + sorted([x[-1] for x in self.finetune 
                           if (x[:-1] == 'layer' and x != 'fc')], 
                          reverse=True)
                )
        else:
            self.finetune = []
            self.tuning_layers = ''
        
        # Load pretrained model
        self.resnet = model_fn(weights=weights)
        
        # Replace final layer
        last_layer_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(last_layer_in, num_classes)
        
        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze specified layers
        for ft_layer in self.finetune:
            if verbose: 
                print(f" turn on gradients for layer {ft_layer}")
            layer = getattr(self.resnet, ft_layer)
            
            for param in layer.named_parameters():
                param[1].requires_grad = True
                if verbose:
                    print(f" param : {param[0]:35s}  Shape: {param[1].shape}  "
                          f"Requires Grad Set to True", end='')
                
    @abstractmethod
    def _get_model_config(self):
        """Return (model_fn, model_name, default_weights) tuple."""
        pass
        
    def forward(self, x):
        return self.resnet(x)    

class PretrainedResNet18(PretrainedResNet):
    def _get_model_config(self):
        return (
            models.resnet18,
            "resnet18",
            models.ResNet18_Weights.IMAGENET1K_V1
        )

class PretrainedResNet34(PretrainedResNet):
    def _get_model_config(self):
        return (
            models.resnet34,
            "resnet34",
            models.ResNet34_Weights.IMAGENET1K_V1
        )

class PretrainedResNet50(PretrainedResNet):
    def _get_model_config(self):
        return (
            models.resnet50,
            "resnet50",
            models.ResNet50_Weights.IMAGENET1K_V2
        )
        
class PretrainedResNet101(PretrainedResNet):
    def _get_model_config(self):
        return (
            models.resnet101,
            "resnet101",
            models.ResNet101_Weights.IMAGENET1K_V2
        )
        
        

class pretrained_resnet18(nn.Module):
    
    def __init__(self, num_classes=13, 
                 finetune = ["fc"],
                 weights=models.ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        assert "fc" in finetune, " 'fc' must be in fine tuning "
        self.name = "resnet18"
        self.finetune =finetune
        self.tuning_layers = ''.join(['fc']+sorted([x[-1] for x in self.finetune if (x[:-1]=='layer' and x != 'fc')], reverse=True))

        self.resnet = models.resnet18(weights=weights)
        last_layer_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(last_layer_in, num_classes)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for ft_layer in finetune:
            print(f" turn on gradients for layer {ft_layer}")
            layer = getattr(self.resnet, ft_layer)
            print(layer)
            print()

            for param in layer.named_parameters():
                print(f" param : {param[0]:35s}  Shape: {param[1].shape}  Requires Grad: {param[1].requires_grad}",end ='')
                param[1].requires_grad = True
                print(f"  ---> {param[1].requires_grad}")
        
    def forward(self, x):
        return self.resnet(x)

class pretrained_resnet34(nn.Module):
    
    def __init__(self, num_classes=13, 
                 finetune = ['fc'],
                 weights=models.ResNet34_Weights.IMAGENET1K_V1):
        super().__init__()
        self.name = "resnet34"
        self.finetune =finetune
        self.tuning_layers = ''.join(['fc']+sorted([x[-1] for x in self.finetune if (x[:-1]=='layer' and x != 'fc')], reverse=True))

        self.resnet = models.resnet34(weights=weights)
        last_layer_in = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Linear(last_layer_in, num_classes)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for ft_layer in finetune:
            print(f" turn on gradients for layer {ft_layer}")
            layer = getattr(self.resnet, ft_layer)
            print(layer)
            print()

            for param in layer.named_parameters():
                print(f" param : {param[0]:35s}  Shape: {param[1].shape}  Requires Grad: {param[1].requires_grad}",end ='')
                param[1].requires_grad = True
                print(f"  ---> {param[1].requires_grad}")
            
    def forward(self, x):
        return self.resnet(x)        


class  pretrained_resnet50(nn.Module):
    
    def __init__(self,num_classes=13, 
                 finetune = ['fc'],
                 weights=models.ResNet50_Weights.IMAGENET1K_V2):
    
        super().__init__()
        self.name = "resnet50"
        self.finetune =finetune        
        self.tuning_layers = ''.join(['fc']+sorted([x[-1] for x in self.finetune if (x[:-1]=='layer' and x != 'fc')], reverse=True))
        
        # init a pretrained resnet
        self.resnet = models.resnet50(weights=weights)
        last_layer_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(last_layer_in, num_classes)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for ft_layer in finetune:
            print(f" turn on gradients for layer {ft_layer}")
            layer = getattr(self.resnet, ft_layer)
            print(layer)
            print()

            for param in layer.named_parameters():
                print(f" param : {param[0]:35s}  Shape: {param[1].shape}  Requires Grad: {param[1].requires_grad}",end ='')
                param[1].requires_grad = True
                print(f"  ---> {param[1].requires_grad}")
            
    def forward(self, x):
        x = self.resnet(x)
        return x


class  pretrained_resnet101(nn.Module):
    
    def __init__(self,num_classes=13, 
                 finetune = ['fc'],
                 weights = models.ResNet101_Weights.IMAGENET1K_V2):
        super().__init__()
        self.name = "resnet101"
        self.finetune =finetune        
        self.tuning_layers = ''.join(['fc']+sorted([x[-1] for x in self.finetune if (x[:-1]=='layer' and x != 'fc')], reverse=True))
        
        # init a pretrained resnet
        self.resnet = models.resnet101(weights=weights)
        last_layer_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(last_layer_in, num_classes)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for ft_layer in finetune:
            print(f" turn on gradients for layer {ft_layer}")
            layer = getattr(self.resnet, ft_layer)
            print(layer)
            print()

            for param in layer.named_parameters():
                print(f" param : {param[0]:35s}  Shape: {param[1].shape}  Requires Grad: {param[1].requires_grad}",end ='')
                param[1].requires_grad = True
                print(f"  ---> {param[1].requires_grad}")
            
    def forward(self, x):
        x = self.resnet(x)
        return x


class ImagenetTransferLearning(L.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        ## Extract all layers prior to the fully connected layer
        layers = list(backbone.children())[:-1]
        
        ## call those layers "feature_extractor"
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
 
        x = self.classifier(representations)
 

class TransferLearner(L.LightningModule):
    def __init__(
        self,
        model_function=None,
        finetune = [],
        num_classes=13,
        weights=None,
        fine_tune_start=99,
        learning_rate=0.001,
    ):
        super().__init__()
        self.t_acc, self.t_loss = 0.0 , 0.0
        self.v_acc, self.t_loss = 0.0 , 0.0
        
        self.save_hyperparameters()
        # print(getattr(models, self.hparams.resnet_model_name)())
        # resnet = getattr(models, self.hparams.resnet_model_name)(
        #     weights=self.hparams.weights
        # )
        self.model = model_function(finetune=finetune)
  
        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="micro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="micro")

    def forward(self, x):
        """
        Run data through the model 
        """
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        """
        Params:

        batch – The output of your data iterable, normally a DataLoader.
        batch_idx – The index of this batch.
        dataloader_idx – The index of the dataloader that produced this batch. (only if multiple dataloaders used)
        """
        # get data and labels from batch
        data, target = batch

        # get prediction
        output = self(data)

        # # calculate batch loss
        loss = F.cross_entropy(output, target)

        # # Batch Predictions.
        pred_batch = output.detach().argmax(dim=1)

        self.t_loss = self.mean_train_loss(loss, weight=data.shape[0])
        self.t_acc = self.mean_train_acc(pred_batch, target)

        # Arguments such as on_epoch, on_step and logger are set automatically depending on
        # hook methods it's been called from
        # self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=True)

        # logging and adding current batch_acc to progress_bar
        # self.log("train/batch_acc", self.mean_train_acc, prog_bar=True, logger=True)

        # return loss
        pass
    
    def on_train_epoch_end(self):
        """Calculate epoch level metrics for the train set"""
        self.log("train/loss", self.mean_train_loss, prog_bar=True, logger=True)
        self.log("train/acc", self.mean_train_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)
        pass
    
    def validation_step(self, batch, *args, **kwargs):

        # get data and labels from batch
        data, target = batch

        # get prediction
        output = self(data)

        # calculate loss
        loss = F.cross_entropy(output, target)

        # # Batch Predictions.
        pred_batch = output.argmax(dim=1)

        # # Update logs.
        self.v_loss = self.mean_valid_loss(loss, weight=data.shape[0])
        self.v_acc  = self.mean_valid_acc(pred_batch, target)

    def on_validation_epoch_end(self):
        # """Calculate epoch level metrics for the validation set"""
        print(f" step:  {self.current_epoch:4d}   train acc: {self.t_acc*100:.3f}  loss: {self.t_loss:.6f}"
              f"    Val acc: {self.v_acc*100:.3f}  loss: {self.v_loss:.6f}")
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True, logger=True)
        self.log("valid/acc", self.mean_valid_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)
        pass
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)    
 