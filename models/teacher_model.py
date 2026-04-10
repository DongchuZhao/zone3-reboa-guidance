import torch
from torch import nn

class TeacherModel(nn.Module):
    def __init__(self, demo_dim, out_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1,16,5,2,2), nn.ReLU(inplace=True),
            nn.Conv3d(16,32,3,2,1), nn.ReLU(inplace=True),
            nn.Conv3d(32,64,3,2,1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.demo = nn.Sequential(nn.Linear(demo_dim,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64+32,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,out_dim))

    def forward(self, ct, demo):
        x = self.backbone(ct).flatten(1)
        d = self.demo(demo)
        return self.head(torch.cat([x,d], dim=1))
