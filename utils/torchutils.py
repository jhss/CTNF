import torch
import torch.nn.functional as F
import torch.distributions as tdists
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch.autograd import Variable
import utils
import sys
import numpy as np

def stable_softmax(preds):
    max_preds = torch.max(preds, dim = 1, keepdim = True)[0]
    p = torch.exp(preds - max_preds)
    if torch.isnan(p).any() == True:
        print("[DEBUG] NaN value is detected in exp of stable_softmax")
        print("preds: ", preds)
        print("max_preds: ", max_preds)
        print("p: ", p)
        sys.exit()
    p = p / p.sum(dim = 1, keepdim = True)
    
    if torch.isnan(p).any() == True:
        print("[DEBUG] NaN value is detected in division of stable_softmax")
        print("p: ", p)
        sys.exit()
    
    return p

def entropy(preds):
    eps = 10e-9
    return -np.sum(preds * np.log(preds + eps), axis = 1)

def wasserstein_example(x, y, lamb, max_lr0, model):
    
    z_hat = x.data.clone().cuda()
    z_hat = Variable(z_hat, requires_grad = True)
    
    optimizer_zt = torch.optim.Adam([z_hat], lr = max_lr0)
    loss_phi = 0
    rho = 0
    #T_adv = 15
    T_adv = 5

    for n in range(T_adv):
        optimizer_zt.zero_grad()
        delta = z_hat - x
        rho = torch.mean((torch.norm(delta.view(len(z_hat), -1), 2, 1)**2))
        latents = model.encoder(z_hat)
        gamma, sldj = model.flows(latents)
        dirichlet, logpsz = model.surnorm(gamma)
        
        base_log_prob = model.log_prob(dirichlet, y)
        nll = -torch.mean(base_log_prob + sldj + logpsz)
        loss = - (nll - lamb * rho)
        loss.backward()
        optimizer_zt.step()

    return z_hat.detach().clone()
