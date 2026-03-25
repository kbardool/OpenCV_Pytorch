
import argparse
import os
import gc
import sys 
import tomllib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TVF


from torch.utils.tensorboard import SummaryWriter
from training_code import (Metrics, SystemConfiguration, TrainingConfiguration, \
                           init_kaiming, main, save_checkpoint, wandb_init)
from models_code import PretrainedResNet50
from dataloader_code import KenyanFoodDataModule

pd.set_option('display.width',180)
torch.set_printoptions(linewidth=180)
np.set_printoptions(linewidth=180)


for p in ['../..']:
    if p not in sys.path:
        # print(f"insert {p}")
        sys.path.insert(0, p)

# print(sys.path)

def delete_vars(variables = ['model',  'metrics', 'train_config', 'sys_config', 'optimizer', 'scheduler', 'train_loader', 'val_loader', 'data_module']):
    for vr in variables:
        global_vars = globals()
        if vr in global_vars:
            try:
                del globals()[vr]
                print(f" {vr:15s} deleted . . . ")
            except Exception as e:
                print(e)
        else:
            print(f" {vr:15s} NOT defined . . .")

    print(f" gc.collect() : {gc.collect()}")
    print(f" Cuda empty cache : {torch.cuda.empty_cache()}")

#----------------------------------------------------
# Parameters 
#----------------------------------------------------
timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
logger = logging.getLogger(__name__) 
logLevel = os.environ.get('LOG_LEVEL', 'INFO').upper()
FORMAT = '%(asctime)s - %(name)s - %(levelname)s: - %(message)s'
logging.basicConfig(level="INFO", format= FORMAT)

logger.info(f" Execution started : {timestamp} ")
logger.info(f" Pytorch version  : {torch.__version__}  \t\t Number of threads: {torch.get_num_threads()}")
logger.info(f" WandB version    : {wandb.__version__}  \t\t Pandas version: {pd.__version__}  ")

parser = argparse.ArgumentParser(description="Training script for Kenyan Food Classification")
parser.add_argument("--config","-c", type=str, dest="configuration", required=True, help=" yaml file containing hyperparameters to use")
args = parser.parse_args()

print(f"Running with configuration file: {args.configuration}")
# delete_vars()
gc.collect()

#----------------------------------------------------
# Configurations and Metrics
#----------------------------------------------------
metrics = Metrics()
sys_config = SystemConfiguration()
train_config = TrainingConfiguration()

with open(args.configuration, "rb") as f:
    input_params = tomllib.load(f)

for k,v in input_params.items():
    k = k.lower()
    # print(f" {k:25s}  {type(v)}  {v:}    {data.get(k, v)}")
    if k not in train_config.__dict__.keys():
        print(f" '{k}'  not found in training configuration - added . . . ")
    setattr(train_config, k, v)
    
train_config.class_boost   = np.array(train_config.class_boost) if train_config.class_boost else None
train_config.session_name  = os.path.basename(__file__)

#----------------------------------------------------
# Data Loader
#----------------------------------------------------
data_module = KenyanFoodDataModule(
    data_root = train_config.data_root, 
    batch_size = train_config.batch_size, 
    num_workers = train_config.num_workers, 
    image_shape = train_config.image_shape,
    class_boost = train_config.class_boost,
    augmentation = train_config.augmentation,
    pin_memory=True,seed=84
)
data_module.prepare_data()
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

#----------------------------------------------------
# Model 
#----------------------------------------------------
# model = PretrainedResNet34(finetune=train_config.tuning_layers, name = train_config.model_name)
# model = PretrainedResNet101(finetune=train_config.tuning_layers)
# model = pretrained_resnet50(finetune=train_config.tuning_layers)
model = PretrainedResNet50(finetune=train_config.tuning_layers)
if train_config.initialization is not None:
    init_kaiming(model, distribution=train_config.initialization)
else:
    print(" No initialization applied . . . ")
    
#---------------------------------------------------------
# Generate run name
#---------------------------------------------------------
train_config.model_name = model.name
do = f"_p{train_config.dropout_rate:3.1f}" if train_config.dropout_rate != 0.0 else ""
bs = f"BS_{train_config.batch_size:03d}"
lr = f"LR_{ train_config.init_learning_rate:.1e}"
wd = f"wd{train_config.weight_decay:.1e}"
wp = f"wp_{train_config.warmup_iters:02d}"
train_config.run_name = f"{train_config.model_name}_{bs}_{lr}_{model.tuning_layers}_{wd}_{wp}"

#---------------------------------------------------------
# tensorboard summary writer
#---------------------------------------------------------
if train_config.use_tensorboard:
    tb_writer = SummaryWriter(
        log_dir = os.path.join(train_config.logs_root, train_config.run_name),
        comment = "week 7 Training"
    )

#---------------------------------------------------------
# optimizers
#---------------------------------------------------------
match train_config.optimizer:
    case "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr= train_config.init_learning_rate,
                              momentum=0.9,
                              dampening=0,
                              weight_decay=train_config.weight_decay,
                              nesterov=True)
    case 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr= train_config.init_learning_rate,
                               weight_decay=train_config.weight_decay)
    case _:
        print(f"Unknown optimizer type: {train_config.optimizer}")

#---------------------------------------------------------
# schedulers
#---------------------------------------------------------

# constant_scheduler = optim.lr_scheduler.ConstantLR(optimizer, 
#                                                  factor=0.5,  # Start at 10% of base LR
#                                                  total_iters=4     # Warmup for 5 epochs
#                                                )
#
# multistep_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2,4,6,8], gamma=0.25)
#
# poly_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, 
#                                                  power=1.0, 
#                                                  total_iters=10)

match train_config.scheduler:
    case "ReduceLROnPlateau":
        scheduler  = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=train_config.lr_factor,
                                                  patience=train_config.lr_patience, 
                                                  cooldown=train_config.lr_cooldown, 
                                                  threshold=0.00001, 
                                                  threshold_mode='rel', 
                                                  min_lr=0, eps=1e-08)
    case "Cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs_count - train_config.warmup_iters)
    case _:
        print(f"Unknown scheduler type: {train_config.scheduler}")

if train_config.scheduler == "Cosine":    
    start_factor = 1.0 if train_config.warmup_iters == 0 else 0.1
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, 
                                             start_factor=start_factor,   # Start at 10% of base LR 
                                             total_iters=train_config.warmup_iters    # Warmup for 5 epochs
                                             # train_config.init_learning_rate/train_config.warmup_iters,  
                                            ) 

    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[train_config.warmup_iters]  # Switch after epoch 5
    )

wandb_session = wandb_init(train_config)
wandb_session.watch(model, log_freq=100)

train_config.run_name += f"_{wandb_session.id}" if wandb_session else ''

print(" Training Configuration : ")
for k,v in train_config.__dict__.items():
    print(f"  {k:25s}: {v}")

model, metrics = main(model, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      system_configuration=sys_config,
                      train_config=train_config, 
                      metrics = metrics,
                      train_loader = train_loader,
                      val_loader  = val_loader,
                      wandb_session = wandb_session,
                      tb_writer=None
                    )

wandb_session.finish()

## Save Model 
model_file_name =f"{train_config.run_name}_aug_{data_module.augmentation}"+ \
                 f"_wd{train_config.weight_decay:.1e}_final_ep{train_config.last_epoch}_{wandb_session.id}.pt"
logger.info(f"Save model to {model_file_name}")

save_checkpoint(model, optimizer, scheduler, metrics, train_config, 
               model_dir='models', model_file_name=model_file_name)

logger.info(f" Execution ended : {datetime.now().strftime('%Y_%m_%d_%H:%M:%S')} ")
