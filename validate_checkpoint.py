import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

from tqdm import tqdm

FOOD101_CLASSES = 101
DATASET_PATH = 'dataset'
CHECKPOINT_URL = 'https://huggingface.co/AlexKoff88/mobilenet_v2_food101/resolve/main/pytorch_model.bin'

def fix_names(state_dict):
    state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
    return state_dict

def load_checkpoint(model):  
    #'https://github.com/AlexKoff88/mobilenetv2_food101/raw/48a0362e617f6fb06b9519037fa1d1ddfbbc77fd/checkpoints/pytorch_model.bin'
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, progress=False)
    weights = fix_names(checkpoint['state_dict'])
    model.load_state_dict(weights)
    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.cpu()
    target = target.cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(model, val_loader):
    predictions = []
    references = []
    
    with torch.no_grad():
        for images, target in tqdm(val_loader):
            output = model(images)
    
            predictions.append(output)
            references.append(target)

            break

        predictions = torch.cat(predictions, axis=0)
        references = torch.cat(references, axis=0)
        
        print(torch.argmax(predictions, axis=1))
        print(references)

        metric = accuracy(output, target)
        print(f'Accuracy @ top1: {metric}')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

val_dataset = datasets.Food101(
    root=DATASET_PATH,
    split = 'test', 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
    download = True
)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, num_workers=4, shuffle=False)

model = models.mobilenet_v2(num_classes=FOOD101_CLASSES) 
model.eval()
model = load_checkpoint(model)

validate(model, val_loader)


   
