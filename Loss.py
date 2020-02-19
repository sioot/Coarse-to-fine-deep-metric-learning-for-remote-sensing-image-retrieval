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


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, margin_push, size_average=True):
        distance_positive = (anchor[0] - positive[0]).pow(2).sum(0)  # .pow(.5)
        distance_negative = (anchor[0] - negative[0]).pow(2).sum(0)  # .pow(.5)
        loss_n1 = torch.max(
            torch.tensor([0, distance_positive - distance_negative + margin_push], requires_grad=True).cuda())
        losses = loss_n1
        return losses.sum()

class New_tripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, margin_push, size_average=True):
        distance_positive = (anchor[0] - positive[0]).pow(2).sum(0)  # .pow(.5)
        distance_negative = (anchor[0] - negative[0]).pow(2).sum(0)  # .pow(.5)
        loss_p1 = distance_positive
        losses = loss_p1 + 10 * loss_n1
        return losses.sum()

class Contrastive_Loss(nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()

    def forward(self, anchor, positive, negative, margin_push, size_average=True):
        distance_positive = (anchor[0] - positive[0]).pow(2).sum(0)  # .pow(.5)
        distance_negative = (anchor[0] - negative[0]).pow(2).sum(0)  # .pow(.5)
        loss_p1 = distance_positive
        losses = loss_p1
        return losses.sum()

class log_ratio(nn.Module):
    def __init__(self):
        super(tri_ratio, self).__init__()

    def forward(self, anchor, i_out, j_out, gt1, gt2, gt3, size_average=True):
        distance_a_i = (anchor[0] - i_out[0]).pow(2).sum(0)  # .pow(.5)
        distance_a_j = (anchor[0] - j_out[0]).pow(2).sum(0)  # .pow(.5)

        gt1 = torch.tensor((1-gt1)).pow(1/2)
        gt2 = torch.tensor((1-gt2)).pow(1/2)
        A = (torch.log(distance_a_i/distance_a_j) - torch.log(gt1/gt2)).pow(2).sum(0)
        loss = A
        return loss.sum()

class tri_ratio(nn.Module):
    def __init__(self):
        super(tri_ratio, self).__init__()

    def forward(self, anchor, i_out, j_out, gt1, gt2, gt3, size_average=True):
        distance_a_i = (anchor[0] - i_out[0]).pow(2).sum(0)  # .pow(.5)
        distance_a_j = (anchor[0] - j_out[0]).pow(2).sum(0)  # .pow(.5)
        distance_i_j = (i_out[0] - j_out[0]).pow(2).sum(0)


        gt1 = torch.tensor((1-gt1)).pow(1/2)
        gt2 = torch.tensor((1-gt2)).pow(1/2)
        gt3 = torch.tensor((1-gt3)).pow(1/2)



        A = (torch.log(distance_a_i/distance_a_j) - torch.log(gt1/gt2)).pow(2).sum(0)
        B = (torch.log(distance_a_i/distance_i_j) - torch.log(gt1/gt3)).pow(2).sum(0)
        C = (torch.log(distance_a_j/distance_i_j) - torch.log(gt2/gt3)).pow(2).sum(0)
        loss = (1/3)*(A + B + C)
        return loss.sum()