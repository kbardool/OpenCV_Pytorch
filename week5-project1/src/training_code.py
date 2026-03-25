#!/usr/bin/env python
# coding: utf-8

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import wandb 
from torchvision import datasets, transforms


def image_preprocess_transforms():

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

    return preprocess


def image_common_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    """
    pre-process + normalization
    """
    preprocess = image_preprocess_transforms()

    common_transforms = transforms.Compose([
        preprocess,
        transforms.Normalize(mean, std)
    ])

    return common_transforms


# def data_augmentation_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
#     """
#     random_augmentations -> pre-process (resize/crop/ToTensor) --> normalization
#     """
#     preprocess = image_preprocess_transforms()

#     augmentation_transforms = transforms.Compose([
#         preprocess,
#         transforms.Normalize(mean, std)
#     ])

#     return common_transforms

from torchvision.transforms import v2

def data_augmentation_transforms(mean=(0.4611, 0.4359, 0.3905), 
                                 std=(0.2193, 0.2150, 0.2109)):
    
    augmentation_options=[
                          v2.ElasticTransform(alpha=100.0, sigma = 5.0),
                          v2.RandomResizedCrop(size=(256, 256), scale=(0.0,1.0)),
                          v2.RandomHorizontalFlip(p=0.5),
                          v2.RandomVerticalFlip(p=0.5),
                          v2.RandomRotation(degrees=65),
                          v2.AugMix(severity= 5),
                         ]
    preprocess = image_preprocess_transforms()
    
    _transforms = v2.Compose([
        transforms.RandomChoice(transforms=augmentation_options), 
        preprocess,
        transforms.Normalize(mean, std)
    ])
    
    return _transforms
    

def init_model(m):
    for l in m.body:
        if isinstance(l, torch.nn.modules.Conv2d):
            # print(f" Convolutional layer: {l.weight.shape}   {l.bias.shape}")
            # print(f" Initialize kernel")
            init.kaiming_uniform_(l.weight, nonlinearity='relu') # Example: Kaiming for ReLU
            if l.bias is not None:
                # print(f" Initialize Bias")
                init.zeros_(l.bias) # Biases are typically set to zero    
    print(f" {m.name} - model trunk initialized")
    
    for l in m.head:
        if isinstance(l, torch.nn.modules.Linear):
            # print(f" Linearlayer: {l.weight.shape}   {l.bias.shape}")
            # print(f" Initialize kernel")
            init.kaiming_uniform_(l.weight, nonlinearity='relu') # Example: Kaiming for ReLU
            if l.bias is not None:
                # print(f" Initialize Bias")
                init.zeros_(l.bias) # Biases a
    print(f" {m.name} - model head initialized")


def get_mean_std(data_root, num_workers=4):

    transform = image_preprocess_transforms()

    loader = data_loader(data_root, transform)

    batch_mean = torch.zeros(3)
    batch_mean_sqrd = torch.zeros(3)

    for batch_data, _ in loader:
        batch_mean += batch_data.mean(dim=(0, 2, 3)) # E[batch_i] 
        batch_mean_sqrd += (batch_data ** 2).mean(dim=(0, 2, 3)) #  E[batch_i**2]

    # E[dataset] = E[E[batch_1], E[batch_2], ...]
    mean = batch_mean / len(loader)

    # var[X] = E[X**2] - E[X]**2

    # E[X**2] = E[E[batch_1**2], E[batch_2**2], ...]
    # E[X]**2 = E[E[batch_1], E[batch_2], ...] ** 2

    var = (batch_mean_sqrd / len(loader)) - (mean ** 2)

    std = var ** 0.5
    # print('mean: {}, std: {}'.format(mean, std))

    return mean, std



def data_loader(data_root, transform, batch_size=16, shuffle=False, num_workers=2, subset_size = 1.0):
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    dataset = torch.utils.data.Subset(dataset,np.arange(0,len(dataset),1./subset_size).astype(int))
    print(f" SubsetSize: {subset_size:f}   Dataset size: {len(dataset)}")
    
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle)

    return loader



def get_data(batch_size, data_root, num_workers=4, data_augmentation=False, subset_size = 1.0):
    """
    setup pytorch data loaders for training and validation data 
    """
    train_data_path = os.path.join(data_root, 'training')
    valid_data_path = os.path.join(data_root, 'validation')

    mean, std = get_mean_std(data_root=train_data_root, num_workers=num_workers)
    # mean = torch.tensor([0.4610, 0.4347, 0.3897]) 
    # std = torch.tensor([0.2734, 0.2641, 0.2616])
    common_transforms = image_common_transforms(mean, std)

    if data_augmentation:
        data_transformations = data_augmentation_transforms(mean, std)
    else:
        data_transformations = common_transforms

    train_loader = data_loader(train_data_path,
                               transform=data_transformations,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers, 
                               subset_size = subset_size)

    valid_loader = data_loader(valid_data_path,
                               transform=common_transforms,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers, 
                               subset_size=subset_size)

    return train_loader,valid_loader


 
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
    batch_size: int = 4      # 10 
    epochs_count: int = 20    # 50  
    init_learning_rate: float = 0.001  ## 0.1  # initial learning rate for lr scheduler
    log_interval: int = 5  
    test_interval: int = 1  
    data_root: str = "./cat-dog-panda" 
    num_workers: int = 2  
    device: str = 'cuda'  
    model_name:str = 'unknown'
    subset_size: float = 1.00
    dropout_rate:float = 0.0
    run_name: str =""


class Metrics():
    def __init__(self):
        self.best_loss_ep = -1
        self.best_loss = torch.tensor(np.inf)
        self.best_loss_acc = 0.0
        self.best_acc = 0.0

        # epoch train/test loss
        self.train_loss = np.array([])
        self.val_loss = np.array([])
        self.test_loss = np.array([])

        # epch train/test accuracy
        self.train_acc = np.array([])
        self.val_acc = np.array([]) 
        self.test_acc = np.array([]) 


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

    # change model in training mood
    model.train()

    # to get batch loss
    batch_loss = np.array([])

    # to get batch accuracy
    batch_acc = np.array([])

    for batch_idx, (data, target) in enumerate(train_loader):

        # clone target
        indx_target = target.clone()
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

        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        optimizer.step()

        batch_loss = np.append(batch_loss, [loss.item()])

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]  

        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()

        # accuracy
        acc = float(correct) / float(len(data))

        batch_acc = np.append(batch_acc, [acc])

    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()

    # print('Epoch: {} \nTrain Loss: {:.6f} Acc: {:.4f}'.format(epoch_idx, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc



def validate(train_config: TrainingConfiguration,
                model: nn.Module,
                test_loader: torch.utils.data.DataLoader,
            ) -> float:
    # 
    model.eval()
    test_loss = 0
    count_corect_predictions = 0
    for data, target in test_loader:
        indx_target = target.clone()
        data = data.to(train_config.device)

        target = target.to(train_config.device)

        with torch.no_grad():
            output = model(data)

        # add loss for each mini batch
        test_loss += F.cross_entropy(output, target).item()

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = prob.data.max(dim=1)[1] 

        # add correct prediction count
        count_corect_predictions += pred.cpu().eq(indx_target).sum()

    # average over number of mini-batches
    test_loss = test_loss / len(test_loader)  

    # average over number of dataset
  
    accuracy = ( count_corect_predictions / len(test_loader.dataset)).item()

    # print(type(accuracy) ))
    return test_loss, accuracy




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

 

def main(model, 
         optimizer, 
            scheduler=None, 
            system_configuration=None, 
            train_config=None, 
            metrics = None,
            # data_augmentation=True,
            # subset_size =1.0,
            train_loader = None, 
            test_loader = None,
            wandb_session = None
        ):

    # system configuration
    setup_system(system_configuration)

    # batch size
    batch_size_to_set = train_config.batch_size
    # num_workers
    num_workers_to_set = train_config.num_workers
    # epochs
    epoch_num_to_set = train_config.epochs_count


    # if GPU is available use training config, 
    # else lowers batch_size, num_workers and epochs count
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f" moding model to cuda")
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 4
 
    # send model to device (GPU/CPU)
    model.to(train_config.device)

    # Calculate Initial Test Loss
    init_val_loss, init_val_accuracy = validate(train_config, model, test_loader)
    print(f" Model run: {train_config.run_name}")
    print(f" Model: {train_config.model_name}   Batch Size: {train_config.batch_size}   "
          f"Init LR: {train_config.init_learning_rate}  Epochs in this run: {train_config.epochs_count}")
    print(" Initial Test Loss : {:.6f} \t Initial Test Accuracy : {:.3f} %\n".format(init_val_loss, 
                                                                                   init_val_accuracy*100))

    # trainig time measurement
    print(" Epoch     Train Loss      Train Acc           Test Loss       Test Acc         Elpsd(s)   (s/epoch)   (s/batch)   ETA (s)      LR \n")
    t_begin = time.time()
    for epoch in range(train_config.epochs_count):

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
            current_loss, current_accuracy = validate(train_config, model, test_loader)

            metrics.val_loss = np.append(metrics.val_loss, [current_loss])
            metrics.val_acc  = np.append(metrics.val_acc, [current_accuracy])

            print(" {:3d}   {:14.6f}         {:.4f}      {:14.6f}         {:.4f}     {:10.2f}     {:7.2f}    {:7.2f}     {:7.1f}     {:.1e}".format(
                    epoch, train_loss, train_acc, current_loss, current_accuracy, elapsed_time, speed_epoch, speed_batch, eta, optimizer.param_groups[0]['lr']), end='')
 
            if current_accuracy > metrics.best_acc:
                metrics.best_acc = current_accuracy
                metrics.best_acc_loss = current_loss
                metrics.best_acc_ep = epoch
                print(f"   Acc Improved: {metrics.best_acc*100:.2f}% .", end="")
                save_model(model, device=train_config.device, 
                           model_file_name=train_config.run_name+"_bestacc.pt")
                
            if current_loss < metrics.best_loss:
                metrics.best_loss = current_loss
                metrics.best_loss_acc = current_accuracy
                metrics.best_loss_ep = epoch
                print(f"   Loss Improved: {metrics.best_loss:.6f} .")
                save_model(model, device=train_config.device, 
                           model_file_name=train_config.run_name+"_bestloss.pt")
            else:
                print()

        # Scheduler step
        # print(type(epoch_test_loss), epoch_test_loss.shape)
        scheduler.step(current_loss)
        wandb_session.log({'trn_loss': train_loss, 'trn_acc': train_acc, 
                           'val_loss':current_loss, 'val_acc' : current_accuracy, 
                           'best_loss':metrics.best_loss, 'best_acc':metrics.best_acc,
                           'best_loss_ep':metrics.best_loss_ep, 'best_acc_ep': metrics.best_acc_ep})
    print()
    print("-"*100)    
    print("  Total time: {:.2f}  \n" \
          "  Best Loss Epoch: {:4d}     Best Loss:   {:.6f}      Best Loss Accuracy: {:.3f} % \n" \
          "  Best Acc  Epoch: {:4d}     Best Accuracy: {:.3f} %    Best Accuracy Loss: {:.6f}".format(
             time.time() - t_begin, 
             metrics.best_loss_ep, metrics.best_loss, metrics.best_loss_acc*100,
             metrics.best_acc_ep , metrics.best_acc * 100, metrics.best_acc_loss))
    print("-"*100)    

    return model, metrics ##  epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc, best_loss




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
    plt.title(f"{trn_config.model_name}-p:{trn_config.dropout_rate:.1f} (BS: {trn_config.batch_size}  LR: {trn_config.init_learning_rate:.2e}) - Loss ")

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
    plt.title(f"{trn_config.model_name}-p:{trn_config.dropout_rate:.1f} (BS: {trn_config.batch_size}  LR: {trn_config.init_learning_rate:.2e}) - Acc ")

    fig.savefig(f"{trn_config.model_name}_bs{trn_config.batch_size}_lr{trn_config.init_learning_rate}_sample_loss_acc_plot.png")
    plt.show()

    return   


# ## <font style="color:blue">4.8. Define Models</font>
# 
# Next, define the CNN model. Keep iterating. Do this by training various models. Just ,change the :
#     
# - number of layers
# - parameters inside the layers
# - different types of layers

# In[25]:

##-------------------------------------------------------------------------------------------------------
##
##-------------------------------------------------------------------------------------------------------
# class MyModel(nn.Module):
#     YOUR CODE HERE


def prediction(model, device, batch_input):

    data = batch_input.to(device)

    with torch.no_grad():
        output = model(data)

    # Score to probability using softmax
    prob = F.softmax(output, dim=1)

    # get the max probability
    pred_prob = prob.data.max(dim=1)[0]

    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]

    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()



def get_sample_prediction(model, data_root, mean, std):
    batch_size = 15

    if torch.cuda.is_available():
        device = "cuda"
        num_workers = 8
    else:
        device = "cpu"
        num_workers = 2

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




def wandb_init(args ):
    """
    args : args to log
        args.exp_id
        args.exp_name
        args.exp_description
        args.project_name
    log_metrics: Metrics to log
    """
    print(args)
    log_metrics = []
    verbose=False
    # if wandb.run is not None:
    #     print(f" End in-flight wandb run . . .")
    #     wandb.finish()
    # else:
    #     print(f" Initiate new W&B job run")
    
    # opt['exp_id'] = wandb.util.generate_id()
    # print_dbg(f"{opt['exp_id']}, {opt['exp_name']}, {opt['project_name']}", verbose) 
    if args.exp_id is None:
        wandb_run = wandb.init(project=args.project_name, 
                            entity="kbardool", 
                            name = args.exp_name,
                            notes = args.exp_description,
                            resume="never" )
    else:
        wandb_run = wandb.init(project=args.project_name, 
                            entity="kbardool", 
                            id = args.exp_id, 
                            resume="must" )
        args.exp_id = wandb_run.id
        
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
    
    # wandb.config.update(opt,allow_val_change=True)   ## wandb.config = opt.copy()
    
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


def wandb_log_metrics(data, step = None, commit = True):
    wandb.log( data = data, step = step, commit = commit) 
    return
    

def wandb_end():
    wandb.finish()


def load_configuration(input_params):

    with open(input_params.configuration) as f:
        _args = yaml.safe_load(f)

    input_params = vars(input_params)
    for k,v in input_params.items():
        logger.info(f" command line param {k:25s} : [{v}]")
        if v is not None:
            _args[k] = v

    _args.setdefault('use_prim_optimizer', True)
    _args.setdefault('use_prim_scheduler', True)
    _args.setdefault('use_temp_optimizer', False)
    _args.setdefault('use_annealing', False)
    _args.setdefault('use_sum', False)
    _args.setdefault('SGD_momentum', 0)

    if _args['runmode'] == 'baseline':
        _args['use_single_loss'] = True
        ttl_pfx = 's'
    else:
        ttl_pfx = 'd'

    opt_ttl = 'DualOpt' if (_args['use_temp_optimizer'] ) else 'SnglOpt'

    _args['exp_title'] = _args['exp_title'].format(ttl_pfx = ttl_pfx,
                                                   cpb = _args['cpb'],
                                                   ltnt = _args['code_units'],
                                                   hidden_1 = _args['hidden_1'])

    _args['exp_description'] = _args['exp_description'].format(runmode = _args['runmode'],
                                                               optimizers = opt_ttl,
                                                               ltnt = _args['code_units'],
                                                               hidden_1 = _args['hidden_1'],
                                                               cpb = _args['cpb'],)

    _args['cellpainting_args']['compounds_per_batch'] = _args['cpb']
    _args['ckpt'] = input_params['ckpt']
    _args['batch_size'] = _args['cellpainting_args']['batch_size']


    if _args['ckpt'] is None:
        _args['exp_date'] = datetime.now().strftime('%Y%m%d_%H%M')
        _args['exp_name'] = f"AE_{_args['exp_date']}"
    else:
        ckpt_parse = _args['ckpt'].split('_')
        print(ckpt_parse)
        _args['exp_date'] = ckpt_parse[5]+'_' + ckpt_parse[6]
        _args['exp_name'] = ckpt_parse[0]+'_' + _args['exp_date']

    _args['use_temp_scheduler'] = _args.setdefault('use_temp_scheduler', False) if _args['use_temp_optimizer'] else False


    assert not (_args['use_annealing'] and _args['use_temp_optimizer']), " Temperature annealing and Temp optimization are mutually exclusive"

    logger.info(f" command line param {'exp_title':25s} : [{_args['exp_title']}]")
    logger.info(f" command line param {'exp_description':25s} : [{_args['exp_description']}]")

    # ars = types.SimpleNamespace(**args, **(vars(input_params)))
    return SimpleNamespace(**_args)


def setup_wandb(args, verbose = False):
    if verbose:
        logger.info(f"WANDB_ACTIVE parameter                : {args.WANDB_ACTIVE}")
        logger.info(f"Project Name     (wandb_run.project)  : {args.project_name}")
        logger.info(f"Experiment Id    (wandb_run.id)       : {args.exp_id}")
        logger.info(f"Experiment Title (wandb_run.title)    : {args.exp_title}")
        logger.info(f"Experiment Notes (wandb_run.notes)    : {args.exp_description}")
        logger.info(f"Initial Exp Name (wandb_run.name)     : {args.exp_name}")
        logger.info(f"Initial Exp Date (extract from name)  : {args.exp_date}")

    if args.WANDB_ACTIVE:
        wandb_status = "***** Initialize NEW  W&B Run *****" if args.exp_id is None else "***** Resume EXISTING W&B Run *****" 
        logger.info(f"{wandb_status}")

        wandb_run = init_wandb(args)
        args.exp_id = wandb_run.id
        args.exp_date = args.exp_name[3:]
        logger.info(f" Experiment Name  : {args.exp_name}")
        logger.info(f" Experiment Date  : {args.exp_date}")
    else: 
        wandb_status = "***** W&B Logging INACTIVE *****"
        wandb_run = None

    logger.info(f"{wandb_status}")
    logger.info(f"WANDB_ACTIVE     : {args.WANDB_ACTIVE}")
    logger.info(f"Project Name     : {args.project_name}")
    logger.info(f"Experiment Id    : {args.exp_id}")
    logger.info(f"Experiment Name  : {args.exp_name}")
    logger.info(f"Experiment Date  : {args.exp_date}")
    logger.info(f"Experiment Title : {args.exp_title}")
    logger.info(f"Experiment Notes : {args.exp_description}")
    return wandb_run


# import wandb
# import random

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="test-project",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()












