import torch.nn as nn
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ImageMLP(nn.Module):
    def __init__(self):
        super(ImageMLP,self).__init__()
        self.Linear= nn.Sequential(
            nn.Linear(2,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3), 
            nn.Sigmoid()
        ).to(device)
    
    def forward(self,x):
        out=self.Linear(x)
        return out

class Fourier_MLP(nn.Module):
    def __init__(self):
        super(Fourier_MLP,self).__init__()
        self.Linear= nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3), 
            nn.Sigmoid()
        ).to(device)
    
    def forward(self,x):
        out=self.Linear(x)
        return out

class RadonMLP(nn.Module):
    def __init__(self):
        super(ImageMLP,self).__init__()
        self.Linear= nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        out=self.Linear(x)
        return out
    
class NLRadon(nn.Module):
    def __init__(self):
        super(ImageMLP,self).__init__()
        self.Linear= nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,3),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        out=self.Linear(x)
        return out

