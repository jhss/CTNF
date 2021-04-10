import numpy as np
import torch
import torch.nn as nn
import os
import sys
import argparse
sys.path.append(".")
from collections import OrderedDict
from tqdm import tqdm

from model import flows as fnn
from model.ctnf import CTNF
from utils.metrics_lib import expected_calibration_error_multiclass as ece
from data.data_loader import *
from utils.calMetric import *

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torchvision import transforms as tvt
from torchvision import datasets as tdatasets
import torch.nn.functional as F

from utils.torchutils import wasserstein_example
# Why doesn't this work?
#from utils.metrics_lib import expected_calibration_error_multiclass as ece

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser()
parser.add_argument('--ood', 
                    default = 'fmnist',
                    help = 'OOD dataset (fmnist, emnist, nmnist)')
args = parser.parse_args()

# MNIST loader
train_loader, valid_loader = get_loader(dataset = 'mnist',
                                        path = '/home/jh/Documents/Research/Datasets',
                                        bsz = 32)

# Three type of corruptions exist (speckle_noise, pixelate, contrast)
corruption_loader = get_loader(dataset = 'mnist_r',
                               path = "/home/jh/Documents/Research/Datasets/",
                               bsz = 500)

# Three type of OOD exist ()
ood_loader = get_loader(dataset = args.ood,
                        path = '/home/jh/Documents/Research/Datasets',
                        bsz = 500)

base_type = 'dirichlet'

ctnf = CTNF(encoder_type = 'lenet',
            n_class = 10, 
            n_flow_blocks = 8, 
            n_flow_hidden = 64,
            base_dist = base_type,
            alphas = [6.0, 0.5])

ctnf.load_encoder('./pretrained_models/mnist_encoder_50.pth')
ctnf.load_flow('./pretrained_models/mnist_flow.pth')
ctnf.load_surnorm("./pretrained_models/mnist_surnorm.pth")
ctnf.encoder.eval()

# optimize MAF and SurNorm, respectively.
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

probs = []
labels = []
valid_probs = []
ood_probs = []
cnt = 0
total = 0

for val_batch in valid_loader:
    x = val_batch[0].to(device)
    x.requires_grad = True
    y = val_batch[1].to(device)

    with torch.no_grad():
        latents = ctnf.encoder(x)
        gamma, sldj = ctnf.flows(latents)
        dirichlet, logpsz = ctnf.surnorm(gamma)

    preds = ctnf.predict(dirichlet, sldj + logpsz)
    preds = preds.to(device)
    valid_probs.append(preds)

    preds_idx = torch.max(preds, dim = 1)[1].cuda()
    cnt += sum(y == preds_idx)
    total += y.shape[0]

    batch = gamma.shape[0]
    for pred, label in zip(preds, y):
        probs.append(pred.tolist())
        labels.append(label.item())
    
val_score = cnt / float(total)
probs = np.asarray(probs)
labels = np.asarray(labels)
valid_entropy = -np.sum(probs * np.log(probs + 10e-9), axis = 1)
print("Validation Accuracy: {:.3f}, ece: {:.3f}".format(val_score, 
                                                        ece(np.asarray(probs), np.asarray(labels))))

for k in range(5):
    for i, batch in enumerate(corruption_loader[k]):
        cnt, total = (0, 0)
        probs = []
        labels = []
        x = torch.as_tensor(batch[0], device = device, dtype = torch.float32)
        y = torch.as_tensor(batch[1], device = device, dtype = torch.int64)
        
        with torch.no_grad():
            latents = ctnf.encoder(x)
            gamma, sldj = ctnf.flows(latents)
            dirichlet, logpsz = ctnf.surnorm(gamma)
        preds = ctnf.predict(dirichlet, sldj+logpsz)

        preds = preds.to(device)
        preds_idx = torch.max(preds, dim = 1)[1]
        
        cnt += sum(y == preds_idx)
        total += y.shape[0]

        for pred, label in zip(preds, y):
            probs.append(pred.tolist())
            labels.append(label.item())

        val_c_score = cnt / float(total)
        calibration_error = ece(np.asarray(probs), np.asarray(labels))
        print("Corruption {} ACC: {:.3f} ECE: {:.3f}"\
               .format(k+1, val_c_score, ece(np.asarray(probs), np.asarray(labels))))
        break

for i, batch in enumerate(ood_loader):

    x = batch[0].to(device)
    if args.ood == 'nmnist':
        x = x[:,0,:,:].unsqueeze(1)
    
    with torch.no_grad():
        latents = ctnf.encoder(x)
        gamma, sldj = ctnf.flows(latents)
        dirichlet, logpsz = ctnf.surnorm(gamma)

    if torch.isnan(gamma).any() == True:
        print("[DEBUG] NaN value is detected in svhn_gamma")
        sys.exit()

    preds = ctnf.predict(dirichlet, sldj+logpsz)
    if torch.isnan(preds).any() == True:
        print("[DEBUG] NaN value is detected in svhn_preds")
        sys.exit()
    ood_probs.append(preds)
    preds = preds.to(device)

    if i == 3: break

valid_probs = torch.cat(valid_probs)
ood_probs = torch.cat(ood_probs)
valid_probs = valid_probs.detach().cpu().numpy()
ood_probs   = ood_probs.detach().cpu().numpy()
valid_max   = np.max(valid_probs, axis = 1)
ood_max     = np.max(ood_probs, axis = 1)

print("TPR95: ", tpr95(valid_max, ood_max))
print("AUROC: ", auroc(valid_max, ood_max))
print("AUPR-Out: ", auprOut(valid_max, ood_max))
print("AUPR-In: ", auprIn(valid_max, ood_max))
