import torch
from torch import nn
from torchvision import models 


class ResNet(nn.Module):
    def __init__(self, net_type='resnet50', is_trained=True, num_classes=1):
        super(ResNet, self).__init__()
        self.net = getattr(models, net_type)(pretrained = is_trained)
        
        # get num_features at last layer
        kernel_count = self.net.fc.in_features

        # get backbone of model
        self.base = nn.Sequential(*list(self.net.children())[:-1]) # -1 mean don't get the final layer
        
        # Alternative final layer
        self.classification = nn.Sequential(nn.Linear(in_features = kernel_count, out_features = num_classes))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.base(x).view(x.size()[0], -1)
        x = self.classification(x)
        x = self.sigmoid(x)
         
        return x