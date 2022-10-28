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
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, progress=False)
    weights = fix_names(checkpoint['state_dict'])
    model.load_state_dict(weights)
    return model

def validate(model, val_loader):
    predictions = []
    references = []
    
    with torch.no_grad():
        for images, target in tqdm(val_loader):
            output = model(images)
    
            predictions.append(np.argmax(output, axis=1))
            references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)  

    return accuracy_score(predictions, references)

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

top1 = validate(model, val_loader)

print(f'Accuracy @ top1: {top1}')


   
