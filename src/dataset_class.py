import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import json
import argparse
# import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import time
from torchvision.datasets import CocoDetection
from PIL import Image
import pandas as pd
import sys
sys.path.insert(0, '../src/data/cocoapi/PythonAPI')
from pycocotools.coco import COCO

# Source: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
class cocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, target_transform=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.target_transform = target_transform

    
    def __getitem__(self, index):
        # Coco File
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id for img from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: coco annotations for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image with PIL
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        
        # number of objects in the image
        num_objs = len(coco_annotation)
        
        # Extracting bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, xmax, ymax]
        # In pytorch, input should be [xmin, ymin, xmax, ymax]
        boxes = []
        # Labels for each objet
        labels = []
        # Area of each bounding box
        areas = []
        # IsCrowd: whether or not the object is a crowd of objects
        iscrowd = []
        xy=[]
        subimages=[]
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            bbox=np.array([xmin, ymin, xmax, ymax]).astype(int)
            boxes.append(bbox)
            labels.append(coco_annotation[i]['category_id'])
            areas.append(coco_annotation[i]['area'])
            iscrowd.append(coco_annotation[i]['iscrowd'])
            xy.append([xmin+(xmax-xmin)/2,ymin+(ymax-ymin)/2])
            subimages.append(img.crop(bbox))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.float32)
        
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation['boxes'] = boxes
        my_annotation['labels'] = labels
        my_annotation['image_id'] = img_id
        my_annotation['area'] = areas
        my_annotation['iscrowd'] = iscrowd
        
        imgs=[img]
        
        if self.transforms is not None:
            imgs = [self.transforms(img) for img in subimages]
        if self.target_transform is not None:
            labels = [self.target_transform(label) for label in labels]
#         if self.
        labels = torch.as_tensor(labels, dtype=torch.int64)
            
        imgs=torch.stack(imgs)
        
        xy=torch.tensor(xy)
            
        return imgs, xy, torch.LongTensor(labels), np.array([img_id]*len(labels))
    
    def __len__(self):
        return len(self.ids)

    
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(size=(128, 128)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch):
    imgs=torch.cat([item[0] for item in batch],dim=0)
    xy=torch.cat([item[1] for item in batch],dim=0)
    y=torch.cat([item[2] for item in batch]).flatten()
    img_ids = np.hstack([item[3] for item in batch]).flatten()
    return [imgs,xy,y, img_ids]



        