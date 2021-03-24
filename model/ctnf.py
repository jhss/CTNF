import torch
import torch.nn as nn
import flows
from SupContrast.networks.resnet_big import SupConResNet

class CTNF(nn.Module):
    def __init__(self, n_class, # the number of classes
                 n_flow_blocks, # the number of blocks in flow
                 n_flow_hidden, # the number of hidden nodes in flow
                 base_dist # type of base distribution in flow (Dirichlet or Gaussian Mixture)
                 alphas, # concentration parameters of the Dirichlet distribution
                )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flow_modules = []

        for _ in range(n_blocks):
            flow_modules += [
                flows.MADE(n_class, n_flow_hidden, None, act = 'sigmoid'),
                flows.BatchNormFlow(n_class),
                flows.Reverse(n_class)
                
            ]
        
        flow_modules += [flows.normalization
        self.encoder = SupConResNet().to(device)
        self.flows   = flows.FlowSequential(flow_modules, base_dist).to(device)
        self.pi      = nn.Sequential(nn.Linear(10, 1), nn.ReLU()).to(device)

    def forward(self, x):
        
        latents     = self.encoder(x)
        gamma, sldj = self.flows(latents)

        return 
