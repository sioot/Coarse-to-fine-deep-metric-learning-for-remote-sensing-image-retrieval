from __future__ import print_function
from __future__ import division
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def initialize_model(model_name, feature_extract, use_pretrained):  # True
    model_ft = None
    input_size = 0

    if model_name == "resnet_triple":
        model_ft = models.resnet34(pretrained=use_pretrained)
        model_ft.avgpool = Flatten()
        model_ft.fc = nn.Linear(25088, 4096)
        model_ft.fc2 = nn.Linear(4096, 30)
        modules = list(model_ft.children())
        modules = modules[:-3]
        model_ft = nn.Sequential(*modules)
        model_ft.add_module('1*1_averagepooling', nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1)))
        model_ft.Flat = Flatten()
        model_ft.fc = nn.Linear(3136, 512)
        model_ft.load_state_dict(checkpoint)
        input_size = 224

    # output : [1, 2254]
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size