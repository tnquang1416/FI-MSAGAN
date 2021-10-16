'''
Created on Jun 7, 2021

@see: copied from https://github.com/martkartasev/sepconv/blob/master/src/loss.py
'''

import torch
import torchvision
from torch import nn


class VggLoss(nn.Module):

    def __init__(self):
        super(VggLoss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 (-10)
            * list(model.features.children())[:-10]
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        outputFeatures = self.features(output)
        targetFeatures = self.features(target)

        loss = torch.norm(outputFeatures - targetFeatures, 2)

        return loss
