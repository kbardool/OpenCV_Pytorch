import os
import sys

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from .utils import resize, random_flip, read_img_labels
from .encoder import DataEncoder
import glob
 

class ListDataset(Dataset):
    def __init__(self, root_dir, list_dir, classes , mode, transform, input_size):
        '''
        Args:
          root_dir: (str) directory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root_dir = root_dir
        self.list_dir = list_dir
        self.classes = classes
        self.mode = mode
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []
        self.labels_names = []

        filelist = glob.iglob("*.jpg", root_dir = root_dir)
        for fname in filelist:
            img_boxes, img_labels, img_label_names = read_img_labels(fname, self.list_dir)
            self.fnames.append(fname)
            self.boxes.append(img_boxes)
            self.labels.append(img_labels)
            self.labels_names.append(img_label_names)
        print(f" Images loaded (self.fnames): {len(self.fnames)}")
        self.num_samples = len(self.fnames)
        
    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        path = os.path.join(self.root_dir, self.fnames[idx])
        img = cv2.imread(path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)
        
        img = img[..., ::-1] ## BGR --> RBG

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        label_names = self.labels_names[idx]
        size = self.input_size

        # Resize & Flip
        img, boxes = resize(img, boxes, (size, size))
        if self.mode == 'train':
            img, boxes = random_flip(img, boxes)
        # Data augmentation.
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, boxes, labels, label_names

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        label_names = [x[3] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, w, h)
        encoder = DataEncoder((w, h))
        loc_targets = []
        cls_targets = []
        lbl_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = encoder.encode(boxes[i], labels[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            lbl_targets.append(label_names[i])
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), lbl_targets

    def __len__(self):
        return self.num_samples