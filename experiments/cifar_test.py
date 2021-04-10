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
from utils.calMetric import *
from data.data_loader import *

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
parser.add_argument('--ood', default = 'svhn', help = 'OOD dataset (svhn, lsun, tiny)')
parser.add_argument('--valid_data_path', default = '/home/jh/Documents/Research/Datasets')
parser.add_argument('--c_data_path', 
                    default = '/home/jh/Documents/Research/Datasets/CIFAR-10-C',
                    help = 'Corruption data path (CIFAR-10-C)')
parser.add_argument('--ood_data_path', 
                    default = '/home/jh/Documents/Research/Datasets/',
                    help = 'OOD data path (SVHN, LSUN, Tiny ImageNet)')
args = parser.parse_args()

# CIFAR-10 loader
train_loader, valid_loader = get_loader(dataset = 'cifar',
                                        path = args.valid_data_path,
                                        bsz = 32)

# Three type of corruptions exist (speckle_noise, pixelate, contrast)
corruption_loader = get_loader(dataset = 'cifar_c',
                               path = args.c_data_path,
                               bsz = 500,
                               cifar_c_type = 'speckle_noise')

# Three type of OOD exist (svhn, lsun, tiny)
ood_loader = get_loader(dataset = args.ood,
                        path = args.ood_data_path,
                        bsz = 500)

base_type = 'dirichlet'

ctnf = CTNF(encoder_type = 'resnet20',
            n_class = 10, 
            n_flow_blocks = 6, 
            n_flow_hidden = 64,
            base_dist = base_type,
            alphas = [7.0, 0.5])

ctnf.load_encoder("./pretrained_models/cifar_encoder_300.pth")
ctnf.load_flow("./pretrained_models/cifar_flow.pth")
ctnf.load_surnorm("./pretrained_models/cifar_surnorm.pth")
#ctnf.encoder.eval()
#ctnf.flows.eval()

# optimize MAF and SurNorm, respectively.
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnt = 0
total = 0
probs = []
labels = []
valid_probs = []
ood_probs = []

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

    preds_idx = torch.max(preds, dim = 1)[1]
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
valid_probs = torch.cat(valid_probs)

print("Validation Accuracy: {:.3f}, ece: {:.3f}".format(val_score,
                                                        ece(np.asarray(probs), np.asarray(labels))))

for i, batch in enumerate(corruption_loader):
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

    #preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)
    preds = preds.to(device)
    preds_idx = torch.max(preds, dim = 1)[1]
    
    cnt += sum(y == preds_idx)
    total += y.shape[0]

    for pred, label in zip(preds, y):
        probs.append(pred.tolist())
        labels.append(label.item())

    val_c_score = cnt / float(total)
    calibration_error = ece(np.asarray(probs), np.asarray(labels))
    print("Corruption {} ACC: {:.3f} ECE: {:.3f}".format(i+1, 
                                                         val_c_score, 
                                                         ece(np.asarray(probs), np.asarray(labels))))
for i, batch in enumerate(ood_loader):
    
    if args.ood == 'svhn':
        x = batch[0].to(device)
    else:
        x = batch.to(device)

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

ood_probs = torch.cat(ood_probs)
valid_probs = valid_probs.detach().cpu().numpy() 
ood_probs   = ood_probs.detach().cpu().numpy()
valid_max = np.max(valid_probs, axis = 1)
ood_max   = np.max(ood_probs, axis = 1)

print("TPR95: ", tpr95(valid_max, ood_max))
print("AUROC: ", auroc(valid_max, ood_max))
print("AUPR-Out: ", auprOut(valid_max, ood_max))
print("AUPR-In: ", auprIn(valid_max, ood_max))
