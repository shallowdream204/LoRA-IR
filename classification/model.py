import torch
import torch.nn as nn
from .clip_encoder import CLIPTower
import torch.nn.functional as F

class CLIPDeRouter(nn.Module):
    def __init__(self, vision_tower, cls_num=3):
        super().__init__()

        self.clip_model = CLIPTower(vision_tower)
        self.clip_model.requires_grad_(False)
        hidden_size = self.clip_model.vision_tower.visual_projection.weight.shape[0]
        self.de_extractor = nn.Sequential(nn.Linear(hidden_size*2,hidden_size),nn.GELU(),nn.Linear(hidden_size,hidden_size//2),nn.GELU())
        self.cls_head = nn.Linear(hidden_size//2, cls_num)

    def forward(self, x):
        #x: b,5,c,h,w
        assert len(x.shape) == 5 and x.shape[1] == 5
        x_low = x[:,0,...]
        x_high = x[:,1:,...]
        y_low = self.clip_model(x_low) #b c
        y_high = self.clip_model(x_high.flatten(0,1)) 
        y_high = y_high.view(-1,4,y_high.shape[1])
        y_high = torch.mean(y_high,dim=1)
        y = torch.cat([y_low,y_high], dim=1)
        de_prior = self.de_extractor(y)
        de_cls = self.cls_head(de_prior)

        return de_cls, de_prior