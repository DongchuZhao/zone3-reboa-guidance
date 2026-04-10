import torch
from torch import nn
import torchvision.models as models

class StudentModel(nn.Module):
    def __init__(self, demo_dim, out_dim, in_channels):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        feat = resnet.fc.in_features
        self.demo = nn.Sequential(nn.Linear(demo_dim,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feat+32,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,out_dim))

    def forward(self, img2d, demo):
        f = self.cnn(img2d).flatten(1)
        d = self.demo(demo)
        return self.head(torch.cat([f,d], dim=1))
