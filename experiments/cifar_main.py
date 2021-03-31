import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append(".")
from collections import OrderedDict
from tqdm import tqdm

from model import resnet
from model import flows as fnn
from model.ctnf import CTNF
from utils.metrics_lib import expected_calibration_error_multiclass as ece
from data.data_loader import *

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torchvision import transforms as tvt
from torchvision import datasets as tdatasets
import torch.nn.functional as F

from utils.torchutils import prediction, eval_log_prob, eval_logpsz, wasserstein_example
# Why doesn't this work?
#from utils.metrics_lib import expected_calibration_error_multiclass as ece

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# CIFAR-10 loader
train_loader, valid_loader = get_loader(dataset = 'cifar',
                                        path = '/home/jh/Documents/Research/Datasets',
                                        bsz = 32)
#print(next(iter(valid_loader)))
#sys.exit()

# Three type of corruptions exist (speckle_noise, pixelate, contrast)
corruption_loader = get_loader(dataset = 'cifar_c',
                               path = "/home/jh/Documents/Research/Datasets/CIFAR-10-C",
                               bsz = 500,
                               cifar_c_type = 'speckle_noise')

# Three type of OOD exist (svhn, lsun, tiny)
ood_loader = get_loader(dataset = 'svhn',
                        path = '/home/jh/Documents/Research/Datasets',
                        bsz = 500)

base_type = 'dirichlet'

ctnf = CTNF(n_class = 10, 
            n_flow_blocks = 6, 
            n_flow_hidden = 64,
            base_dist = base_type,
            alphas = [7.0, 0.5])

ctnf.load_encoder('/home/jh/Documents/Research/CTNF/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_1000_cosine_warm/ckpt_epoch_300.pth')
ctnf.encoder.eval()

# optimize MAF and SurNorm, respectively.
optimizer = optim.Adam(ctnf.flows.parameters(), lr = 0.01)
pi_optim  = optim.Adam(ctnf.surnorm.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2000, gamma = 0.1)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 30
eps = 10e-9
w_adv = False

for epoch in range(epochs):
    for step, (x, y) in tqdm(enumerate(train_loader)):
        x = x.to(device)
        x.requires_grad = True
        y = y.to(device)
        
        # [Step1] Encoding
        with torch.no_grad():
            latents = ctnf.encoder(x)

        # [Step2] Foward to Flow 
        gamma, sldj  = ctnf.flows(latents)
        dirichlet, _ = ctnf.surnorm(gamma) 

        # [Step3] Optimize pi_{\phi}(w)
        
        s_preds = ctnf.surnorm.inverse(dirichlet.detach().clone()).squeeze(1)
        s = torch.sum(gamma, dim = 1).detach().clone()

        pi_optim.zero_grad()
        l2_loss = torch.mean(torch.square(s_preds - s))
        l2_loss.backward()
        pi_optim.step()

        # [Step4] Evaluate negative log-likelihood

        dirichlet, logpsz = ctnf.surnorm(gamma)
        base_log_prob   = ctnf.log_prob(dirichlet, y)
        nll = -torch.mean(base_log_prob + sldj + logpsz)

        # [Step5] Generates W-adv example
        
        if w_adv == True:
            w_img = wasserstein_example(x.detach().clone(), y.detach().clone(),
                                        encoder = encoder, lamb = 1.5, max_lr0 = 0.01,
                                        flow = flow, log_type = p_type,
                                        base_dist = base_dist, s_preds = s_preds)


            if torch.isnan(w_img).any() == True:
                print("[DEBUG] NaN value is detected in W_img")
                sys.exit()

            with torch.no_grad():
                w_latents = encoder(w_img)
            # [Step5] Increase the entropy of W-adv

            w_gamma, w_sldj = flow(w_latents)
            w_gamma /= torch.sum(w_gamma, dim = 1, keepdim = True)
            entropy = torch.sum(w_gamma * torch.log(w_gamma + eps), dim = 1)
            entropy = torch.mean(entropy)
            
            loss = nll + 6.0*entropy
        else:
            loss = nll

        # [Step6] Optimize flow
        ctnf.flows.zero_grad()
        loss.backward()
        clip_grad_norm_(ctnf.flows.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        if (step+1) % 500 == 0 or step == 1497:
            for param in optimizer.param_groups:
                print("Learning Rate: ", param['lr'])
            cnt = 0
            total = 0
            probs = []
            labels = []
            valid_probs = []
            ood_probs = []
            
            '''
            for val_batch in valid_loader:
                x = val_batch[0].to(device)
                x.requires_grad = True
                y = val_batch[1].to(device)

                with torch.no_grad():
                    latents = ctnf.encoder(x)
         
                gamma, sldj = ctnf.flows(latents)
                dirichlet, logpsz = ctnf.surnorm(gamma)
                preds = ctnf.log_prob(dirichlet, sldj + logpsz)
                #preds = prediction(dirichlet, sldj + logpsz, base_type)
                #preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)

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
            print("{}th epoch {}th step, \nValidation Accuracy: {:.3f}, ece: {:.3f}".format(epoch, step, val_score, ece(np.asarray(probs), np.asarray(labels))))
            '''

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
                print("Corruption {} ACC: {:.3f} ECE: {:.3f}"\
                       .format(i+1, val_c_score, ece(np.asarray(probs), np.asarray(labels))))

        '''
            entropies = []
            for i, batch in enumerate(svhn_loader):

                x = batch[0].to(device)
                
                with torch.no_grad():
                    latents = encoder(x)

                gamma, sldj = flow(latents)

                if torch.isnan(gamma).any() == True:
                    print("[DEBUG] NaN value is detected in svhn_gamma")
                    sys.exit()

                preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)
                if torch.isnan(preds).any() == True:
                    print("[DEBUG] NaN value is detected in svhn_preds")
                    sys.exit()
                ood_probs.append(preds)
                preds = preds.to(device)
                entropy = -torch.sum(preds * torch.log(preds + 10e-9), dim = 1)
                entropies.append(entropy)

                if i == 10: break
                           
            entropies = torch.cat(entropies)
            valid_entropy_quantile = np.quantile(valid_entropy, [0, 0.25, 0.5, 0.75, 1])
            ood_entropy_quantile   = np.quantile(entropies.cpu().detach().numpy(), 
                                                 [0, 0.25, 0.5, 0.75, 1])
            if np.isnan(ood_entropy_quantile).any() == True:
                print("[DEBUG] NaN Value is detected in ood_entropy")
                sys.exit()
            print("valid entropy: ", np.quantile(valid_entropy, [0, 0.25, 0.5, 0.75, 1]))
            print("ood entropy: ", np.quantile(entropies.cpu().detach().numpy(), [0, 0.25, 0.5, 0.75, 1]))
            lsun_entropies = []
            lsun_probs = []
            for i, batch in enumerate(lsun_loader):

                x = batch.to(device)
                
                with torch.no_grad():
                    latents = encoder(x)

                gamma, sldj = flow(latents)

                preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)
                ood_probs.append(preds)
                preds = preds.to(device)
                lsun_probs.append(preds)

                entropy = -torch.sum(preds * torch.log(preds + 10e-9), dim = 1)
                lsun_entropies.append(entropy)

                if i == 10: break
            
            tiny_entropies = []
            tiny_probs = []
            for i, batch in enumerate(tiny_loader):

                x = batch.to(device)
                
                with torch.no_grad():
                    latents = encoder(x)

                gamma, sldj = flow(latents)

                preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)
                ood_probs.append(preds)
                preds = preds.to(device)
                tiny_probs.append(preds)

                entropy = -torch.sum(preds * torch.log(preds + 10e-9), dim = 1)
                tiny_entropies.append(entropy)

                if i == 10: break
            
            lsun_entropies = torch.cat(lsun_entropies)
            tiny_entropies = torch.cat(tiny_entropies)
            lsun_probs = torch.cat(lsun_probs)
            tiny_probs = torch.cat(tiny_probs)
            print("lsun entropy: ", np.quantile(lsun_entropies.cpu().detach().numpy(), [0, 0.25, 0.5, 0.75, 1]))
            print("tiny entropy: ", np.quantile(tiny_entropies.cpu().detach().numpy(), [0, 0.25, 0.5, 0.75, 1]))
            ood_probs = torch.cat(ood_probs)
            print(ood_probs.shape)
            '''
        #print("entropy mean: ", torch.mean(entropies))
            #print("entropy[:10]: ", entropies[:10])
#torch.save(flow.state_dict(), os.path.join(flow_path, "adv_b1024_init_e100_L12_epoch{}".format(epoch)))
