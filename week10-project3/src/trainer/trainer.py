"""Unified class to make training pipeline for deep neural networks."""
import os
import datetime

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import wandb
import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .configuration import TrainerConfig
from .hooks import test_hook_default, train_hook_default
from .visualizer import Visualizer
import numpy as np

class Trainer:  # pylint: disable=too-many-instance-attributes
    """ Generic class for training loop.

    Parameters
    ----------
    model : nn.Module
        torch model to train
    loader_train : torch.utils.DataLoader
        train dataset loader.
    loader_test : torch.utils.DataLoader
        test dataset loader
    loss_fn : callable
        loss function
    metric_fn : callable
        evaluation metric function
    optimizer : torch.optim.Optimizer
        Optimizer
    lr_scheduler : torch.optim.LrScheduler
        Learning Rate scheduler
    configuration : TrainerConfiguration
        a set of training process parameters
    data_getter : Callable
        function object to extract input data from the sample prepared by dataloader.
    target_getter : Callable
        function object to extract target data from the sample prepared by dataloader.
    visualizer : Visualizer, optional
        shows metrics values (various backends are possible)
    # """
    def __init__( # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
        trainer_config: TrainerConfig, 
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device: Union[torch.device, str] = "cuda",
        model_save_best: bool = True,
        model_saving_frequency: int = 1,
        save_dir: Union[str, Path] = "checkpoints",
        data_getter: Callable = itemgetter("image"),
        target_getter: Callable = itemgetter("target"),
        stage_progress: bool = True,
        visualizer: Union[Visualizer, None] = None,
        get_key_metric: Callable = itemgetter("top1"),
        wandb_session : wandb.Run = None
    ):
        self.data_getter = data_getter
        self.device = device
        self.get_key_metric = get_key_metric
        self.hooks = {}
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}
        self.metric_fn = metric_fn
        self.model = model
        self.model_save_best = model_save_best
        self.model_saving_frequency = model_saving_frequency
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.stage_progress = stage_progress
        self.target_getter = target_getter
        self.trainer_config = trainer_config
        self.visualizer = visualizer
        self.wandb_session = wandb_session
        self._register_default_hooks()

    def fit(self, epochs):
        """ Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        best_acc = 0.0
        best_loss = np.inf
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        for epoch in iterator:
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter
            )

            
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )
            
            if self.visualizer:
                self.visualizer.update_charts(
                    None, 
                    output_train['loss'], 
                    output_test['metric'], 
                    output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], 
                    epoch
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()
            
            # best_acc = max([self.get_key_metric(item) for item in self.metrics['test_metric']])
            
            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test, self.wandb_session, best_acc)

            wandb_id = f"_{self.wandb_session.id}" if self.wandb_session is not None else "" 
            
            if self.model_save_best:
                current_acc = self.get_key_metric(output_test['metric'])
                current_loss = output_test['loss']
                if current_acc > best_acc:
                    print(f" Epoch: {epoch:3d} - Current Acc: {current_acc:.5f}   Best acc: {best_acc:.5f} - Save Checkpoint")
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, self.trainer_config.run_name) + wandb_id +'_best_acc.pth'
                    )
                    best_acc = current_acc
                    
                if current_loss < best_loss:
                    print(f" Epoch: {epoch:3d} - Current Loss: {current_acc:.5f}   Best Loss: {best_acc:.5f} - Save Checkpoint")
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, self.trainer_config.run_name) + wandb_id +'_best_loss.pth'
                    )
                    best_loss = current_loss
            else:
                if (epoch + 1) % self.model_saving_frequency == 0:
                    print(f" Epoch: {epoch:3d} - Current Acc: {current_acc:.5f}   Best acc: {best_acc:.5f} - Save Checkpoint")
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        # os.path.join(self.save_dir, self.model.__class__.__name__) + '_' +
                        os.path.join(self.save_dir, self.trainer_config.run_name) + wandb_id + '_' +
                        str(datetime.datetime.now()) + '.pth'
                    )

        return self.metrics

    def register_hook(self, hook_type, hook_fn):
        """ Register hook method.

        Arguments:
            hook_type (string): hook type.
            hook_fn (callable): hook function.
        """
        self.hooks[hook_type] = hook_fn

    def _register_default_hooks(self):
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)
