import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RND(nn.Module):
    def __init__(self, obs_dim, hid=256, out_dim=128, lr=1e-3, device="cpu"):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim), nn.ReLU() # 256 * 128 
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(), 
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim), nn.ReLU() # 256 * 128 
        ).to(device)
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr) # Adam optimizer 
        self.device = device 

    @th.no_grad() 
    def novelty(self, obs):  
        tgt = self.target(obs)
        pred = self.predictor(obs)
    
        return ((tgt - pred).pow(2).mean(dim=1, keepdim=True)).detach()

    def update(self, obs):
        tgt = self.target(obs)
        pred = self.predictor(obs)
        loss = F.mse_loss(pred, tgt)
        self.opt.zero_grad() 
        loss.backward()
        self.opt.step() 
        return loss.detach()
