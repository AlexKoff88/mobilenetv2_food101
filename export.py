import sys
import os

import torch
import torch.nn as nn
import torchvision.models as models

FOOD101_CLASSES = 101

def fix_names(state_dict):
    state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
    return state_dict

model = models.mobilenet_v2(num_classes=FOOD101_CLASSES)    

if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        
        checkpoint = torch.load(checkpoint_path)
        weights = fix_names(checkpoint['state_dict'])
        model.load_state_dict(weights)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, checkpoint['epoch']))

dummy_input = torch.randn(1, 3, 224, 224)

input_names = ["input"] 
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "mobilenet_v2_food101.onnx", verbose=True, input_names=input_names, output_names=output_names)