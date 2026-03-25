import os
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import lightning as L
from sklearn.model_selection import train_test_split
from PIL import Image


 #    class counts:        {0: 537, 1: 733, 2: 407, 3: 420, 4: 147, 5: 527, 6: 372, 7: 410, 8: 180, 9: 666, 10: 280, 11: 342, 12: 534}
 #   class weights: tensor([0.0019, 0.0014, 0.0025, 0.0024, 0.0068, 0.0019, 0.0027, 0.0024, 0.0056, 0.0015,  0.0036,  0.0029,  0.0019])
 # boosted weights: tensor([0.0019, 0.0014, 0.0025, 0.0024, 0.0204, 0.0019, 0.0027, 0.0024, 0.0056, 0.0015,  0.0036,  0.0029,  0.0019])

class KenyanFoodDataset(Dataset):
    """
    This custom dataset class takes root directory and train flag, 
    and returns dataset training dataset if train flag is true 
    else it returns validation dataset.
    """
    
    def __init__(self, data_dict, training = True, transform=None):
        
        """
        init method of the class.
        
         Parameters:
         
         data_dict (pandas Dataframe): path of root directory.  
         training (tuple): Data split between training and validation data.
         image_shape (int or tuple or list): [optional] int or tuple or list. Default is None. 
                                             If it is not None image will resize to the given shape.       
         transform (method): method that will take PIL image and transform it.
         
        """
        # set transform attribute
        self.transform = transform
        self.data_dict = data_dict
        self.training = training
        print(f" Dataset initialized with {len(self.data_dict)} samples")
        print(f" Sample data: \n {self.data_dict.head()}")
    def __len__(self):
        """
        return length of the dataset
        """
        return len(self.data_dict)
    
    
    def __getitem__(self, idx):
        """
        For given index, return images with resize and preprocessing.
        """
        image = Image.open(self.data_dict['image_path'][idx]).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        if self.training:
            return image, self.data_dict['label'][idx] 
        else:
            # print(str(self.data_dict['id'][idx]))
            return image, str(self.data_dict['id'][idx])


class KenyanFoodDataModule(L.LightningDataModule):
    
    def __init__(self, 
                 data_root   : str, 
                 batch_size  : int = 12, 
                 num_workers : int = 4, 
                 image_shape : int = 512,
                 shuffle     : bool = False,
                 pin_memory  : bool = False,
                 augmentation: str = None,
                 class_boost : np.array = None, 
                 seed = 42):
        super().__init__()
        print(" DataModule Initialization Started")
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.augmentation = augmentation.lower() if augmentation is not None else None
        self.pin_memory = pin_memory
        self.seed = seed
        
        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)
            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert len(image_shape) == 1 or len(image_shape) == 2, 'Invalid image_shape tuple size'
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError 
        else:
            self.image_shape = image_shape

        self.training_data = pd.read_csv( os.path.join(self.data_root, 'train.csv'), engine='python')
        self.label_names = list(self.training_data['class'].unique())
        self.label_names.sort()
        self.name_to_label = {x:i for i,x in enumerate(self.label_names)}
        self.label_to_name = {i:x for x,i in self.name_to_label.items()}
        if class_boost is not None:
            assert len(class_boost) == len(self.label_names), "class_boost length should be same as number of classes"
            self.class_boost = torch.Tensor(class_boost)
        else:
            self.class_boost = torch.ones(len(self.label_names))

        self._set_augmentation_strategy()

        # print(f" class names  : {self.label_names}")
        print(f" name_to_label: {self.name_to_label}")
        print(f" label to name: {self.label_to_name}")
        print(f" image shape  : {self.image_shape}")
        print(f" batch_size   : {self.batch_size}")
        print(f" class boost  : {self.class_boost}")
        print(f" num_workers  : {self.num_workers}")
        print(f" seed         : {self.seed}")
        print(f" augmentation : {self.augmentation}")
        print(f" training augmentations: \n {self.aug_transforms}")
        print(" DataModule Initialization Completed")
    
    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        """Setup datasets for each stage (fit, validate, test, predict)"""
        
        print(f" DataModule Setup started - stage {stage}")
        #-----------------------------------------------------------------
        # We use ImageNet normalization mean/std when fine-tuning/transfer learning
        #-----------------------------------------------------------------

        self.test_data = pd.read_csv( os.path.join(self.data_root, 'test.csv'), engine='python')
        self.test_data['image_path'] = self.test_data['id'].map(lambda x: f"{os.path.join(self.data_root,str(x))}.jpg")        
        
        self.training_data['label'] = self.training_data['class'].map(lambda x: self.name_to_label[x])
        self.training_data['image_path'] = self.training_data['id'].map(lambda x: f"{os.path.join(self.data_root,str(x))}.jpg")        
        
        self.train_split, self.val_split, y_train, y_val = train_test_split(
                 self.training_data, self.training_data['label'], test_size=0.15, 
                 stratify=self.training_data['label'], random_state=self.seed)

        self.train_split.reset_index(inplace=True)
        self.val_split.reset_index(inplace=True)
        
        ## Create weighted sampler
        
        _class_counts = Counter(self.train_split['label'])
        self.class_counts = { x: _class_counts[x] for x in sorted(_class_counts)}
        self.class_weights = torch.Tensor([1.0/self.class_counts[x] for x in sorted(self.class_counts)])
        print(f"    class counts: {self.class_counts}")
        print(f"   class weights: {self.class_weights}")
        if self.class_boost is not None:
            self.class_weights = self.class_weights * self.class_boost
            print(f" boosted weights: {self.class_weights}")
        
        sample_weights =[self.class_weights[x] for x in self.train_split.label]
        self.weighted_sampler = WeightedRandomSampler(
            weights= sample_weights,
            num_samples=len(self.train_split),
            replacement=True,
            generator = torch.Generator().manual_seed(self.seed)
        )
                
        # train_data_path = os.path.join(self.data_root, "training")
        # val_data_path = os.path.join(self.data_root, "validation")

        self.train_dataset = KenyanFoodDataset(data_dict=self.train_split,
                                                 training=True,
                                                 transform=self.aug_transforms)

        self.val_dataset = KenyanFoodDataset(data_dict=self.val_split,
                                                 training=True,
                                                 transform=self.common_transforms)

        self.test_dataset = KenyanFoodDataset(data_dict=self.test_data,
                                                 training=False,
                                                 transform=self.common_transforms)
        
        # Apply different transforms to each split
        # if stage == "fit" or stage is None:
        #     self.train_dataset = self._apply_transform(train_dataset, train_transforms)
        #     self.val_dataset = self._apply_transform(val_dataset, eval_transforms)
        
        # if stage == "test" or stage is None:
        #     self.test_dataset = self._apply_transform(test_dataset, eval_transforms)
        print(" DataModule Setup completed")

    def _set_augmentation_strategy(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.preprocess_transforms = transforms.Compose(
            [transforms.Resize([x+32 for x in self.image_shape]), 
             transforms.CenterCrop(self.image_shape), 
             transforms.ToTensor(),
            ]
        )

        self.common_transforms = transforms.Compose([
             transforms.Resize([x+32 for x in self.image_shape]), 
             transforms.CenterCrop(self.image_shape), 
             transforms.ToTensor(),
             transforms.Normalize(mean, std)
        ])
        
        match self.augmentation:
            case None:
                self.aug_transforms = self.common_transforms
            case 'v1':
                self.aug_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(self.image_shape, scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=5),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.RandomAffine(degrees = 0, translate=(0.1,0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
            case 'v5':   ## adds RandomErasing to v1
                self.aug_transforms = transforms.Compose([    
                        transforms.RandomResizedCrop(self.image_shape, scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.RandomAffine(degrees = 0, translate=(0.1,0.1)),               
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),     
                        transforms.RandomErasing(p=0.5, scale=(0.02, 0.5), ratio=(0.3,1.2),value=0),
                        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3,1.2),value=0),
                    ])    

            case 'v6':   ## adds AugMix to v1
                self.aug_transforms = transforms.Compose([   
                        transforms.RandomResizedCrop(self.image_shape, scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.RandomAffine(degrees = 0, translate=(0.1,0.1)),            
                        transforms.AugMix(severity= 3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.RandomErasing(p=0.30, scale=(0.02, 0.25), ratio=(0.3,1.2),value=0),
                    ])                            
            case 'v2':  ## Randomly applies one of the augmentations in the list to each image
                augmentation_options=[
                        # transforms.ElasticTransform(alpha=100.0, sigma = 5.0),
                        transforms.RandomResizedCrop(self.image_shape, scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.RandomAffine(degrees = 0, translate=(0.1,0.1)),            
                        # transforms.AugMix(severity= 3),
                        ]
                self.aug_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(self.image_shape, scale=(0.7, 1.0)),
                        transforms.RandomChoice(transforms=augmentation_options), 
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.RandomErasing(p=0.25, scale=(0.02, 0.5), ratio=(0.3,1.2),value=0),
                    ])

            case _:
                raise RuntimeError(' Wrong augmentation version')
        
        return 
                        
    def train_dataloader(self):
        # train loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.weighted_sampler,
            num_workers=self.num_workers,
            pin_memory = self.pin_memory,
            persistent_workers = True,
            drop_last = False
            # shuffle=self.shuffle,
            # in_order = False
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory = self.pin_memory,
            persistent_workers = True,
            drop_last = False
        )
        return val_loader

    def test_dataloader(self):
        # validation loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory = self.pin_memory
        )
        return test_loader      
        
    # def data_augmentation_transforms(mean, std):
    
    #     augmentation_options=[
    #                           v2.ElasticTransform(alpha=100.0, sigma = 5.0),
    #                           v2.RandomResizedCrop(size=(256, 256), scale=(0.0,1.0)),
    #                           v2.RandomHorizontalFlip(p=0.5),
    #                           v2.RandomVerticalFlip(p=0.5),
    #                           v2.RandomRotation(degrees=65),
    #                           v2.AugMix(severity= 5),
    #                          ]
    #     preprocess = image_preprocess_transforms()
        
    #     _transforms = v2.Compose([
    #         transforms.RandomChoice(transforms=augmentation_options), 
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor()
    #         transforms.Normalize(mean, std)
    #     ])
        
    #     return _transforms