#!/usr/bin/env python
# coding: utf-8

import os
import time
import logging
from tqdm.autonotebook import tqdm
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import wandb 
import torchmetrics
from torchvision import datasets, transforms
import torchvision.transforms.functional as TVF
from torchmetrics.classification import (MulticlassConfusionMatrix, 
                                         MulticlassCalibrationError,
                                         MulticlassAccuracy, 
                                         MulticlassAUROC, MulticlassF1Score)

from torchinfo import summary
logger = logging.getLogger(__name__)

# Plot few images
def display_class_images(imgs, lbls = None, names=None, filter= None):
    plt.rcParams["figure.figsize"] = (18, 9)
    plt.figure
    if names is None:
        names = ['']*len(imgs)
    if lbls is None:
        lbls = ['']*len(imgs)
    plt_pos = 0
    for i in range(len(imgs)):
        if lbls[i] and lbls[i]==filter:
            plt.subplot(3, 6, plt_pos+1)
            plt_pos += 1
            # print(f" {type(imgs[i])}  {imgs[i].min():5f}  {imgs[i].max():5f}")
            img = TVF.to_pil_image(imgs[i])
            plt.imshow(img)
            plt.gca().set_title('Tgt: {0} {1}'.format(lbls[i], names[i]), fontsize=9)
    plt.show()


# Plot few images
def display_images(imgs, lbls = None, names=None):
    plt.rcParams["figure.figsize"] = (18, 9)
    plt.figure
    if names is None:
        names = ['']*len(imgs)
    if lbls is None:
        lbls = ['']*len(imgs)
    for i in range(len(imgs)):
        plt.subplot(3, 6, i+1)
        # print(f" {type(imgs[i])}  {imgs[i].min():5f}  {imgs[i].max():5f}")
        img = TVF.to_pil_image(imgs[i])
        plt.imshow(img)
        plt.gca().set_title('Tgt: {0} {1}'.format(lbls[i], names[i]), fontsize=9)
    plt.show()


def display_image(img, lbl = None):
    # plt.rcParams["figure.figsize"] = (15, 9)
    plt.figure(figsize=(4,4))
    lbl = '' if lbl is None else lbl
    if img.ndim == 4 :
        img = img.squeeze()
        print(img.shape)
    img = TVF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
    plt.imshow(img)
    plt.gca().set_title('Target: {0} '.format(lbl))
    plt.show()
        # break


def print_model_summary(model = None, batch_size = 4, image_size =  (3, 224, 224), depth = 3):
    summary_col_names = ("input_size",
                "output_size",
                "num_params",
                # "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable" )

    input_size = [batch_size, *image_size]
    print(f"\nInput size: {input_size}")
    print(summary(model, input_size = input_size, col_names=summary_col_names, depth =depth))


def denormalize(y, mean = None, std = None) :
    mean = mean if mean is not None else torch.tensor([0.485, 0.456, 0.406]) 
    std  = std if std is not None else torch.tensor([0.229, 0.224, 0.225]) 
    match y.ndim:
        case 3:
            return y * std[:,None,None] + mean[:,None,None]
        case 4:
            return y * std[None,:,None,None] + mean[None, :,None,None]

  
@dataclass
class SystemConfiguration:
    '''
    Describes the common system setting needed for reproducible training
    '''
    seed: int = 21  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    augmentation: str = ""
    batch_size: int = 4      # 10 
    class_boost: np.array = None
    data_root: str = "./images" 
    device: str = 'cuda'  
    dropout_rate:float = 0.0
    early_stopping: bool = False
    epochs_count: int = 20    # 50  
    image_shape: int = 224,
    initialization: str = None
    init_learning_rate: float = 0.001  ## 0.1  # initial learning rate for lr scheduler
    last_epoch : int = -1
    log_interval: int = 5  
    logs_root: str = "./logs" 
    lr_factor: float = 1.0   ## LR reduce factor for ReduceLR on Plateau
    model_name:str = 'unknown'
    notebook_name: str = ""
    num_workers: int = 2  
    optimizer: str = ""
    project_name: str =""
    run_name: str =""
    session_name: str = ""
    scheduler: str = ""
    subset_size: float = 1.00
    test_interval: int = 1  
    tuning_layers : str = ""
    use_tensorboard: bool = False
    warmup_iters: int = 0
    weight_decay: float = 1.0e-04

@dataclass
class InferenceConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 18      # 10 
    data_root: str = "./images" 
    device: str = 'cuda'  
    epochs_count: int = 20    # 50  
    image_shape: int = 224,
    initialization: str = None
    logs_root: str = "./logs" 
    model_name:str = 'unknown'
    notebook_name: str = ""
    num_workers: int = 1
    project_name: str =""
    run_name: str =""
    session_name: str = ""
    test_interval: int = 1  

@dataclass
class Metrics():
    '''
    Metrics used in the training process
    '''
    best_loss_ep = -1
    best_loss = torch.tensor(np.inf)
    best_loss_acc = 0.0
    best_acc = 0.0

    # epoch train/test loss
    train_loss = np.array([])
    train_acc  = np.array([])
    # train_acc_2  = np.array([])

    # epch train/test accuracy
    val_loss = np.array([])
    val_acc  = np.array([]) 
    val_acc_macro  = np.array([]) 
    val_cm = np.array([]) 
    val_aucroc = np.array([])
    val_mcce = np.array([])
    
    test_loss = np.array([])
    test_acc  = np.array([]) 

    #f1 metrics
    train_f1 = np.array([])
    val_f1 = np.array([])


def init_kaiming(m, distribution = "kaiming_uniform", mode = "fan_in"):
    layer_list = list(m.named_modules())
    # The code is creating a list called `layer_list` that contains all the named modules within the
    # object `m`.
    print(" initializing model with {} distribution ".format(distribution))
    if distribution == "kaiming_uniform":   
        init_func = init.kaiming_uniform_
    elif distribution == "kaiming_normal":
        init_func = init.kaiming_normal_
    else:
        print(f" distribution {distribution} not supported - model not initialized ")
        return
    print(f" Number of layers/modules to inspect:  {len(layer_list)}")

    for (layer_name, layer) in layer_list:
        print(f" layer name:  {layer_name:30s} -- type:{str(type(layer)).split('\'')[1]} ",end = "")
        if isinstance(layer, nn.modules.Conv2d):
            print(f"\n    Initialize Convolutional layer: weight shape: {layer.weight.shape} ",end = "")
            init_func(layer.weight, mode= mode, nonlinearity='relu') # Example: Kaiming for ReLU
            if layer.bias is not None:
                print("\n    Initialize Bias",end = "")
                init.zeros_(layer.bias) # Biases are typically set to zero    
            print()
        elif isinstance(layer, nn.modules.Linear):
            print(f"\n    Initialize linear layer: {layer.weight.shape}   {layer.bias.shape}",end = "")
            init_func(layer.weight, mode= mode, nonlinearity='relu') # Example: Kaiming for ReLU     
            if layer.bias is not None :
                print("\n    Initialize Bias",end = "")
                init.zeros_(layer.bias) # Biases are typically set to zero    
            print()
        else:
            print(" -- layer not initialized")
    print(" model initialized ")

    
def setup_system(system_config: SystemConfiguration) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def train(train_config: TrainingConfiguration, 
            model: nn.Module, 
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            epoch_idx: int
        ) -> None:
    """
    train the model for one epoch
    
    Args:
        train_config (TrainingConfiguration): training configuration
        model (nn.Module): model to be trained
        optimizer (torch.optim.Optimizer): optimizer to use
        train_loader (torch.utils.data.DataLoader): training data loader
        epoch_idx (int): current epoch index
    
    Returns:
        epoch_loss (float): average loss for the epoch
        epoch_acc (float): average accuracy for the epoch
        trn_acc (float): torchmetrics accuracy for the epoch    
    """
    # change model in training mood
    model.train()

    # to get batch loss
    batch_loss = np.array([])

    batch_acc = torchmetrics.classification.Accuracy(task="multiclass",
                                                     average = 'micro',
                                                     num_classes=13).to(train_config.device)
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (data, target) in enumerate(train_loader):

        # clone target
        # indx_target = target.clone()
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(train_config.device)
        # send target to device
        target = target.to(train_config.device)

        # reset parameters gradient to zero
        optimizer.zero_grad()

        # forward pass to the model
        output = model(data)

        # cross entropy loss
        loss = F.cross_entropy(output, target)
        
        #-----------------------------------------
        # Score to probability using softmax
        # prob = F.softmax(output, dim=1)
        # get the index of the max probability
        # pred = prob.data.max(dim=1)[1]  
        # correct prediction
        # correct = pred.eq(target).sum()       
        # accuracy
        # acc = correct * 1.0 / len(data)
        # acc.requires_grad_()
        # print(pred)
        # print(sum)
        # print(acc)
        #-----------------------------------------
        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        optimizer.step()

        batch_loss = np.append(batch_loss, [loss.item()])
        avg_loss = batch_loss.mean()
        batch_acc.update(output, target)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })        
        progress_bar.update(1)
    
    progress_bar.close()
    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.compute().cpu().item()

    return epoch_loss, epoch_acc



def validate(train_config: TrainingConfiguration,
            model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            metrics: Metrics
            ) -> float:
    """
    validate the model on validation dataset
    Args:
        train_config (TrainingConfiguration): training configuration
        model (nn.Module): model to be validated
        val_loader (torch.utils.data.DataLoader): validation data loader    
    Returns:
        val_loss (float): average validation loss
        accuracy (float): average validation accuracy
        val_acc_2 (float): torchmetrics accuracy for the validation dataset
    """
    model.eval()
    
    _val_loss = 0.0
    # count_corect_predictions = 0
    _val_cm = MulticlassConfusionMatrix(num_classes=13,
                                        normalize=None).to(train_config.device)
    # MulticlassAccuracy:
    #
    # average: micro: Sum statistics over all labels
    #          macro: Calculate statistics for each label and average them
    #       weighted: calculates statistics for each label and computes weighted 
    #                 average using their support    
    _val_acc = MulticlassAccuracy(num_classes=13, 
                                  average="micro").to(train_config.device)
    _val_acc_macro = MulticlassAccuracy(num_classes=13,
                                 average='macro').to(train_config.device)    
    _val_aucroc = MulticlassAUROC(num_classes=13, 
                                  average="weighted", 
                                  thresholds=10).to(train_config.device)
    _val_f1 = MulticlassF1Score(num_classes=13, 
                                average="weighted").to(train_config.device)
    _val_mcce = MulticlassCalibrationError(num_classes = 13).to(train_config.device)
    
    for data, target in val_loader:
        
        data = data.to(train_config.device)
        target = target.to(train_config.device)

        with torch.no_grad():
            preds = model(data)
        # print(f" shapes: {data.shape}   target: {target.shape}  preds: {preds.shape}")
        
        # add loss for each mini batch
        _val_loss += F.cross_entropy(preds, target).item()
        _val_acc.update(preds,target)
        _val_acc_macro.update(preds,target)
        _val_cm.update(preds, target)
        _val_aucroc.update(preds, target)
        _val_f1.update(preds, target)
        _val_mcce.update(preds, target)
        
        # Score to probability using softmax
        # probs = F.softmax(preds, dim=1)
        # # get the index of the max probability
        # pred_labels = probs.data.max(dim=1)[1] 
        # count_corect_predictions += pred_labels.cpu().eq(indx_target).sum()

        
    # average over number of mini-batches
    # accuracy = ( count_corect_predictions / len(val_loader.dataset)).item()
    metrics.val_loss    = np.append(metrics.val_loss  , _val_loss / len(val_loader))
    metrics.val_acc     = np.append(metrics.val_acc   , _val_acc.compute().cpu().numpy())
    metrics.val_acc_macro= np.append(metrics.val_acc_macro   , _val_acc_macro.compute().cpu().numpy())
    metrics.val_cm      = np.append(metrics.val_cm    , _val_cm.compute().cpu().numpy())
    metrics.val_aucroc  = np.append(metrics.val_aucroc, _val_aucroc.compute().cpu().numpy())
    metrics.val_f1      = np.append(metrics.val_f1    , _val_f1.compute().cpu().numpy())
    metrics.val_mcce    = np.append(metrics.val_mcce  , _val_mcce.compute().cpu().numpy())
    val_loss   = _val_loss / len(val_loader)
    val_acc    = _val_acc.compute().cpu().numpy()
    return val_loss, val_acc


def save_model(model, device, model_dir='models', model_file_name='cat_dog_panda_classifier.pt'):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save(model.state_dict(), model_path)

    if device == 'cuda':
        model.to('cuda')
    return


def load_model(model, model_dir='models', model_file_name='cat_dog_panda_classifier.pt'):
    model_path = os.path.join(model_dir, model_file_name)

    # loading the model and getting model parameters by using load_state_dict
    model.load_state_dict(torch.load(model_path))
    return model    


def save_checkpoint(model, optimizer, scheduler, metrics, train_config, 
                   model_dir='models', model_file_name='cat_dog_panda_classifier.pt'):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if train_config.device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save({'epoch':train_config.last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics,
                'training_config':train_config}
                , model_path)
    print(f" model written to {model_path}")
    if train_config.device == 'cuda':
        model.to('cuda')
    return


def load_checkpoint(ckpt_dir='models', ckpt_file_name='cat_dog_panda_classifier.pt'):

    checkpoint_path = os.path.join(ckpt_dir, ckpt_file_name)
    print(checkpoint_path)
    # make sure you transfer the model to cpu.
    print(f" loading checkpoint from {checkpoint_path}")
    return torch.load(checkpoint_path, weights_only=False)


def main(model, 
         optimizer, 
            scheduler=None, 
            system_configuration=None, 
            train_config=None, 
            metrics = None,
            train_loader = None, 
            val_loader = None,
            wandb_session = None,
            tb_writer = None
        ):
    
    # system configuration
    setup_system(system_configuration)

    # if GPU is available use training config, 
    train_config.device = "cuda" if torch.cuda.is_available() else "cpu"
 
    print(f" moving model to {train_config.device}")
    model.to(train_config.device)

    # Calculate Initial Test Loss
    init_val_loss, init_val_acc = validate(train_config, model, val_loader, metrics)
    print(f" Model run: {train_config.run_name}")
    print(f" Model: {train_config.model_name}   Batch Size: {train_config.batch_size}   "
          f"Init LR: { train_config.init_learning_rate}  Epochs in this run: {train_config.epochs_count}")
    print(" Initial Test Loss : {:.6f} \t Initial Test Accuracy : {:.3f} %\n".format(
            init_val_loss, init_val_acc*100))

    # trainig time measurement
    print(" Epoch     Trn Loss    Trn Acc      Val Loss    Val Acc      macro       Elpsd(s)     (s/ep)    (s/bat)    ETA (s)      LR        F1    Calib")
            # 6       0.269182     0.9134      0.802107     0.7533     0.7263         364.22      52.03       0.90     3538.2    9.8e-03   0.7528  0.1000   Acc Imp: 75.33% .   Loss Imp: 0.802107 .    
    t_begin = time.time()
    start_epoch = train_config.last_epoch + 1
    end_epoch = train_config.last_epoch + 1 + train_config.epochs_count

    patience_counter = 0
    
    for epoch in range(start_epoch, end_epoch):
        train_config.last_epoch = epoch
        
        # Train
        train_loss, train_acc = train(train_config, model, optimizer, train_loader, epoch)
        metrics.train_loss = np.append(metrics.train_loss, [train_loss])
        metrics.train_acc = np.append(metrics.train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * train_config.epochs_count - elapsed_time

        # Validate
        if epoch % train_config.test_interval == 0:
            
            val_loss, val_acc = validate(train_config, model, val_loader, metrics)

            print(" {:3d}   {:12.6f}  {:9.4f}  {:12.6f}  {:9.4f}  {:9.4f}     {:10.2f}    {:7.2f}    {:7.2f}    {:7.1f}    {:.1e}   {:.4f}  {:.4f}  ".format(
                    epoch, train_loss, train_acc, val_loss, val_acc, metrics.val_acc_macro[-1], elapsed_time, speed_epoch, speed_batch, 
                    eta, scheduler.get_last_lr()[0], metrics.val_f1[-1], metrics.val_mcce[-1]), end='')
 
            if val_acc > metrics.best_acc:
                metrics.best_acc = val_acc
                metrics.best_acc_loss = val_loss
                metrics.best_acc_ep = epoch
                print(f"Acc: {metrics.best_acc*100:.2f}% ", end="")
                save_model(model, device=train_config.device, 
                           model_file_name=train_config.run_name+"_bestacc.pt")
                
            if val_loss < metrics.best_loss:
                metrics.best_loss = val_loss
                metrics.best_loss_acc = val_acc
                metrics.best_loss_ep = epoch
                print(f"Loss: {metrics.best_loss:.6f} ", end="")
                save_model(model, device=train_config.device, 
                           model_file_name=train_config.run_name+"_bestloss.pt")
                patience_counter = 0 
            else:
                # patience_counter +=1
                pass
            
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                print(f" {scheduler.cooldown_counter:2d}-{scheduler.num_bad_epochs}", end ='')
            else:
                scheduler.step()
        print()
                
        if wandb_session is not None:
            wandb_session.log({'trn_loss': train_loss, 'trn_acc': train_acc, 
                               'val_loss':val_loss, 'val_acc' : val_acc, 
                               'best_loss':metrics.best_loss, 'best_acc':metrics.best_acc,
                               'best_loss_ep':metrics.best_loss_ep, 'best_acc_ep': metrics.best_acc_ep,
                               'lr[0]':scheduler.get_last_lr()[0]})
            
        if tb_writer is not None:
            tb_write_metrics(tb_writer, metrics)
            
        # Early stop based on loss
        if train_config.early_stopping and (patience_counter >= train_config.early_stopping):
            print(f" ***** Early stopping threshold met {train_config.early_stopping} *****")
            print(f" ***** Early stopping at epoch {epoch} *****")
            break
    # end of epoch -------------------
        
    print()
    print("-"*100)    
    print("  Total time: {:.2f}  \n" \
          "  Best Loss Epoch: {:4d}     Best Loss:   {:.6f}      Best Loss Accuracy: {:.3f} % \n" \
          "  Best Acc  Epoch: {:4d}     Best Accuracy: {:.3f} %    Best Accuracy Loss: {:.6f}".format(
             time.time() - t_begin, 
             metrics.best_loss_ep, metrics.best_loss, metrics.best_loss_acc*100,
             metrics.best_acc_ep , metrics.best_acc * 100, metrics.best_acc_loss))
    print("-"*100)    

    return model, metrics 

def tb_write_metrics(tb_writer, mtrcs):
    # add scalar (loss/accuracy) to tensorboard
    tb_writer.add_scalar('Loss/Train', mtrcs.train_loss[-1], epoch)
    tb_writer.add_scalar('Accuracy/Train', mtrcs.train_acc[-1], epoch)
    
    # add scalar (loss/accuracy) to tensorboard
    tb_writer.add_scalar('Loss/Val', mtrcs.val_loss[-1], epoch)
    tb_writer.add_scalar('Accuracy/Val', mtrcs.val_acc[-1], epoch)
    
    # add scalars (loss/accuracy) to tensorboard
    tb_writer.add_scalars('Loss/train-val', {'train': mtrcs.train_loss[-1], 
                                             'validation': mtrcs.val_loss[-1]}, epoch)
    tb_writer.add_scalars('Accuracy/train-val', {'train': mtrcs.train_acc[-1], 
                                                 'validation': mtrcs.val_acc[-1]}, epoch)

def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, 
                       colors=['blue'], 
                       loss_legend_loc='upper center', acc_legend_loc='upper left', 
                       fig_size=(15.0, 5.0),
                       sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2),
                       trn_config = None):

    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()

    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])

    for i in range(len(train_loss)):
        x_train = range(len(train_loss[i]))
        x_val = range(len(val_loss[i]))

        min_train_loss = train_loss[i].min()

        min_val_loss = val_loss[i].min()

        plt.plot(x_train, train_loss[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN LOSS ({0:.4})".format(min_train_loss))
        plt.plot(x_val, val_loss[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID LOSS ({0:.4})".format(min_val_loss))

    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc)
    plt.title(f"{trn_config.model_name}-p:{trn_config.dropout_rate:.1f} (BS: {trn_config.batch_size}  LR: {trn_config.learning_rate:.2e}) - Loss ")

    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])

    for i in range(len(train_acc)):
        x_train = range(len(train_acc[i]))
        x_val = range(len(val_acc[i]))

        max_train_acc = train_acc[i].max() 

        max_val_acc = val_acc[i].max() 

        plt.plot(x_train, train_acc[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN ACC ({0:.4})".format(max_train_acc))
        plt.plot(x_val, val_acc[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID ACC ({0:.4})".format(max_val_acc))

    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc)
    plt.title(f"{trn_config.model_name}-p:{trn_config.dropout_rate:.1f} (BS: {trn_config.batch_size}  LR: {trn_config.learning_rate:.2e}) - Acc ")

    fig.savefig(f"{trn_config.model_name}_bs{trn_config.batch_size}_lr{trn_config.learning_rate}_sample_loss_acc_plot.png")
    plt.show()

    return   



def prediction(model, device, batch_input, max_prob=True):
    """
    get prediction for batch inputs
    """
    # send model to cpu/cuda according to your system configuration
    model.to(device)
    model.eval()

    data = batch_input.to(device)

    output = model(data)

    # get probability score using softmax
    prob = F.softmax(output, dim=1)
    
    if max_prob:
        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]
    else:
        pred_prob = prob.data
    
    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]
    
    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()
    
def get_target_and_prob(model, dataloader, device):
    """
    get targets and prediction probabilities
    """
    
    pred_prob = []
    targets = []
    
    for _, (data, target) in enumerate(dataloader):
        
        _, prob = prediction(model, device, data, max_prob=False)
        
        pred_prob.append(prob)
        
        target = target.numpy()
        targets.append(target)
        
    targets = np.concatenate(targets)
    targets = targets.astype(int)
    pred_prob = np.concatenate(pred_prob, axis=0)
    
    return targets, pred_prob
    
    

def get_sample_prediction(model, data_root, mean, std):
    batch_size = 15

    if torch.cuda.is_available():
        device = "cuda"
        # num_workers = 8
    else:
        device = "cpu"
        # num_workers = 2

    # It is important to do model.eval() before prediction
    model.eval()

    # Send model to cpu/cuda according to your system configuration
    model.to(device)

    # transformed data
    test_dataset_trans = datasets.ImageFolder(root=data_root, transform=image_common_transforms(mean, std))

    # original image dataset
    test_dataset = datasets.ImageFolder(root=data_root, transform=image_preprocess_transforms())

    data_len = test_dataset.__len__()

    interval = int(data_len/batch_size)

    imgs = []
    inputs = []
    targets = []
    for i in range(batch_size):
        index = i * interval
        trans_input, target = test_dataset_trans.__getitem__(index)
        img, _ = test_dataset.__getitem__(index)

        imgs.append(img)
        inputs.append(trans_input)
        targets.append(target)

    inputs = torch.stack(inputs)

    cls, prob = prediction(model, device, batch_input=inputs)

    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (15, 9)
    fig = plt.figure()


    for i, target in enumerate(targets):
        plt.subplot(3, 5, i+1)
        img = transforms.functional.to_pil_image(imgs[i])
        plt.imshow(img)
        plt.gca().set_title('P:{0}({1:.2}), T:{2}'.format(test_dataset.classes[cls[i]], 
                                                     prob[i], 
                                                     test_dataset.classes[targets[i]]))
    fig.savefig('sample_prediction.png')
    plt.show()

    return


def wandb_init(args, verbose = False):
    """
    args : args to log
        args.exp_id
        args.exp_name
        args.exp_description
        args.project_name
        log_metrics: Metrics to log
    """
    log_metrics = []
    verbose=False
 
    # if wandb.run is not None:
    #     print(f" End in-flight wandb run . . .")
    #     wandb.finish()
    # else:
    #     print(f" Initiate new W&B job run")
    settings = wandb.Settings(
        show_errors=True,  # Show error messages in the W&B App
        silent=False,      # Disable all W&B console output
        show_warnings=True, # Show warning messages in the W&B App
        show_info=True, 
        console = "redirect",
        # console_multipart = True,
        # console_chunk_max_seconds = 600,
        notebook_name = args.session_name
    )    
    if not hasattr(args,"wandb_id"):
        # opt['exp_id'] = wandb.util.generate_id()
        # print_dbg(f"{opt['exp_id']}, {opt['exp_name']}, {opt['project_name']}", verbose) 
        wandb_run = wandb.init(project=args.project_name, 
                            entity="kbardool", 
                            name = args.run_name,
                            resume="never" ,
                            settings = settings)
        args.wandb_id = wandb_run.id
    else:
        wandb_run = wandb.init(project=args.project_name, 
                            entity="kbardool", 
                            id = args.wandb_id, 
                            resume="must", 
                            settings = settings)
        args.wandb_id = wandb_run.id
        if wandb_run.project != '':
            args.project_name = wandb_run.project
        else:
            wandb_run.project = args.project_name 
            
        if wandb_run.name != '':
            args.exp_name = wandb_run.name
        else:
            wandb_run.name = args.exp_name 
            
        if wandb_run.notes != '':
            args.exp_description = wandb_run.notes
        else:
            wandb_run.notes = args.exp_description 
        
    wandb.config.update(args,allow_val_change= True)
    
    for metric in log_metrics:
        wandb.define_metric(metric, summary="last")
    # assert wandb.run is None, "Run is still running"
    if verbose:
        print(f" WandB Initialization -----------------------------------------------------------\n"
              f" PROJECT NAME: {wandb_run.project}\n"
              f" RUN ID      : {wandb_run.id} \n"
              f" RUN NAME    : {wandb_run.name}\n"     
              f" RUN NOTES   : {wandb_run.notes}\n"     
              f" --------------------------------------------------------------------------------")
    return wandb_run
 

def wandb_watch(item = None, criterion=None, log = 'all', log_freq = 1000, log_graph = True):
    """
    Note: Increasing the log frequency can result in longer run times
    """
    if item is not None:
        wandb.watch(item,
                    criterion=criterion,
                    log = log,            ###     Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
                    log_freq = log_freq,
                    log_graph = log_graph
                   )        
    return


# def setup_wandb(args, verbose = False):
#     if verbose:
#         logger.info(f"WANDB_ACTIVE parameter                : {args.WANDB_ACTIVE}")
#         logger.info(f"Project Name     (wandb_run.project)  : {args.project_name}")
#         logger.info(f"Experiment Id    (wandb_run.id)       : {args.exp_id}")
#         logger.info(f"Experiment Title (wandb_run.title)    : {args.exp_title}")
#         logger.info(f"Experiment Notes (wandb_run.notes)    : {args.exp_description}")
#         logger.info(f"Initial Exp Name (wandb_run.name)     : {args.exp_name}")
#         logger.info(f"Initial Exp Date (extract from name)  : {args.exp_date}")
# 
#     if args.WANDB_ACTIVE:
#         wandb_status = "***** Initialize NEW  W&B Run *****" if args.exp_id is None else "***** Resume EXISTING W&B Run *****" 
#         logger.info(f"{wandb_status}")
# 
#         wandb_run = wandb_init(args)
#         args.exp_id = wandb_run.id
#         args.exp_date = args.exp_name[3:]
#         logger.info(f" Experiment Name  : {args.exp_name}")
#         logger.info(f" Experiment Date  : {args.exp_date}")
#     else: 
#         wandb_status = "***** W&B Logging INACTIVE *****"
#         wandb_run = None
# 
#     logger.info(f"{wandb_status}")
#     logger.info(f"WANDB_ACTIVE     : {args.WANDB_ACTIVE}")
#     logger.info(f"Project Name     : {args.project_name}")
#     logger.info(f"Experiment Id    : {args.exp_id}")
#     logger.info(f"Experiment Name  : {args.exp_name}")
#     logger.info(f"Experiment Date  : {args.exp_date}")
#     logger.info(f"Experiment Title : {args.exp_title}")
#     logger.info(f"Experiment Notes : {args.exp_description}")
#     return wandb_run


def wandb_log_metrics(data, step = None, commit = True):
    wandb.log( data = data, step = step, commit = commit) 
    return
    
def wandb_end():
    wandb.finish()

 








