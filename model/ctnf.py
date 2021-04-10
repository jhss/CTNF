import torch
import torch.nn as nn
import sys
from collections import OrderedDict
from model import flows
from model.encoder import resnet20, lenet
from torch.distributions.dirichlet import Dirichlet
from utils.torchutils import *

def stable_softmax(preds):
    max_preds = torch.max(preds, dim = 1, keepdim = True)[0]
    p = torch.exp(preds - max_preds)
    p = p / p.sum(dim = 1, keepdim = True)
    return p

class CTNF(nn.Module):
    def __init__(self, 
                 encoder_type, # ['resnet20', 'lenet']
                 n_class, # the number of classes
                 n_flow_blocks, # the number of blocks in flow
                 n_flow_hidden, # the number of hidden nodes in flow
                 base_dist,     # type of base distribution in flow (Dirichlet or Gaussian Mixture)
                 alphas):       # concentration parameters of the Dirichlet distribution
        
        super(CTNF, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flow_modules = []
        base = []

        for _ in range(n_flow_blocks):
            flow_modules += [
                flows.MADE(n_class, n_flow_hidden, None, act = 'sigmoid'),
                flows.BatchNormFlow(n_class),
                flows.Reverse(n_class)
            ]
        if encoder_type == 'resnet20':
            self.encoder = resnet20().to(device)
        elif encoder_type == 'lenet':
            self.encoder = lenet().to(device)

        self.flows   = flows.FlowSequential(flow_modules, base_dist).to(device)
        self.surnorm = flows.SurNorm(n_class).to(device);

        if base_dist == 'dirichlet':
            for i in range(10):
                concentration = alphas[1] * torch.ones(10).to(device)
                concentration[i] = alphas[0]
                base.append(Dirichlet(concentration))
        elif base_dist == 'gmm':
            for i in range(10):
                mean = torch.randn(10).to(device)
                std  = 0.1*torch.ones(10).to(device)
                base.append(MultivariateNormal(mean, scale_tril = torch.diag(std)))

        self.base_dist = base

    def load_encoder(self, path):
        temp_model = torch.load(path)['model']
        new_model = OrderedDict()
        for key, value in temp_model.items():
            new_key = key.split(".")
            new_key = key.replace("module.", "")
            new_key = key.replace("encoder.", "")
            if key[:4] == 'head': continue
            new_model[new_key] = value

        self.encoder.load_state_dict(new_model)

    def load_flow(self, path):
        temp_model = torch.load(path)
        new_model = OrderedDict()
        for key, value in temp_model.items():
            new_key = key.split(".")
            new_key = key.replace("module.", "")
            new_key = key.replace("encoder.", "")
            if key[:4] == 'head': continue
            new_model[new_key] = value

        self.flows.load_state_dict(new_model)
    
    def load_surnorm(self, path):
        temp_model = torch.load(path)
        new_model = OrderedDict()
        for key, value in temp_model.items():
            new_key = key.split(".")
            new_key = key.replace("module.", "")
            new_key = key.replace("encoder.", "")
            if key[:4] == 'head': continue
            new_model[new_key] = value

        self.surnorm.load_state_dict(new_model)
    
    def log_prob(self, rvs, y):
        eps = 10e-9
        batch = rvs.shape[0]
        
        log_probs = []
        
        for i in range(batch):
            log_probs.append(self.base_dist[y[i]].log_prob(rvs[i] + eps))
        log_probs = torch.stack(log_probs)
        return log_probs

    def predict(self, rvs, sldj):
        eps = 10e-9
        batch = rvs.shape[0]
        batch_categorical_dist = torch.zeros(batch, 10)

        for i in range(batch):
            for class_index in range(10):
                base_prob = self.base_dist[class_index].log_prob(rvs[i] + eps)
                batch_categorical_dist[i, class_index] = base_prob + sldj[i]

        batch_categorical_dist = stable_softmax(batch_categorical_dist)

        return batch_categorical_dist

