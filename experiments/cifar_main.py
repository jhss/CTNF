import numpy as np
import torch
import torch.nn as nn
import os
import sys
import flows as fnn

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import OrderedDict
from torch.distributions.gamma import Gamma
import torch.distributions as tdists

from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms as tvt
from torchvision import datasets as tdatasets
from metrics_lib import expected_calibration_error_multiclass as ece
from SupContrast.networks.resnet_big import SupConResNet

import data as data_
import utils
import torch.nn.functional as F

from SupContrast.networks.resnet_big import SupConResNet
from SupContrast.losses import SupConLoss
from utils.torchutils import prediction, eval_log_prob, eval_logpsz, wasserstein_example
from collections import OrderedDict
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# CIFAR-10 loader
train_loader, valid_loader = get_loader(dataset = 'cifar',
                                        path = '/home/jh/Documents/Research/Datasets',
                                        bsz = 32)

# Three type of corruptions exist (speckle_noise, pixelate, contrast)
corruption_loader = get_loader(dataset = 'cifar_c',
                               path = "/home/jh/Documents/Research/Datasets/CIFAR-10-C",
                               bsz = 500,
                               cifar_c_type = 'speckle_noise')

# Three type of OOD exist (svhn, lsun, tiny)
ood_loader = get_loader(dataset = 'svhn',
                        path = '/home/jh/Documents/Research/Datasets',
                        bsz = 500)


p_type = 'dirichlet'
flow_type = 'maf'
alphas = [7.0, 0.5]

if p_type == 'dirichlet':
    base_dist = []
    for i in range(10):
        concentration = alphas[1] * torch.ones(10).cuda()
        concentration[i] = alphas[0]
        base_dist.append(Dirichlet(concentration))
elif p_type == 'gmm':
    base_dist = []
    for i in range(10):
        mean = torch.randn(10).cuda()
        print("mean: ", mean)
        std  = 0.1*torch.ones(10).cuda()
        base_dist.append(MultivariateNormal(mean, scale_tril = torch.diag(std)))

encoder = SupConResNet()
device = 'cuda'

model = torch.load('/home/jh/Documents/Research/CTNF/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_1000_cosine_warm/ckpt_epoch_300.pth')['model']

new_model = OrderedDict()
for key, value in model.items():
    new_key = key.split(".")
    new_key = key.replace("module.", "")
    new_model[new_key] = value

encoder.load_state_dict(new_model)
encoder = encoder.to(device)
cnt = 0
for param in encoder.parameters():
    cnt += param.numel()
print("number of encoder parameters: ", cnt)

encoder.eval()

pi = nn.Sequential(nn.Linear(10, 1), nn.ReLU()).cuda()

if flow_type == 'maf':
    modules = []
    n_blocks = 6
    num_inputs = 10
    num_hidden = 64

    for _ in range(n_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, None, act = 'sigmoid'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

    flow = fnn.FlowSequential(modules, p_type).to(device)

cnt = 0
for param in flow.parameters():
    cnt += param.numel()
print("number of flow parameters: ", cnt)
optimizer = optim.Adam(flow.parameters(), lr = 0.01)
pi_optim  = optim.Adam(pi.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2000, gamma = 0.1)
epochs = 30

iters = 3
eps = 0.15
gamma_eps = 10e-9
w_adv = True
cnt = 0
s_cnt = 0

dro = 'dro' if w_adv == True else 'normal'


for epoch in range(epochs):
    for step, (x, y) in tqdm(enumerate(train_loader)):
        x = x.to(device)
        x.requires_grad = True
        y = y.to(device)
        
        # [Step1] Encoding
        with torch.no_grad():
            latents = encoder(x)

        # [Step2] Foward to Flow 
        gamma, sldj = flow(latents)

        batch = latents.shape[0]

        s = torch.sum(gamma, dim = 1).detach().clone()
        dirichlet = gamma.detach() / s.reshape(-1, 1)

        pi_optim.zero_grad()
        s_preds   = pi(dirichlet).squeeze(1)
        l2_loss = torch.mean(torch.square(s_preds - s))
        l2_loss.backward()
        pi_optim.step()
        
        s_preds = pi(dirichlet).squeeze(1).detach().clone()
        logpsz = eval_logpsz(gamma, s_preds)
        #sum_gamma = torch.sum(gamma, dim = 1)
        #s_preds = pi(dirichlet).squeeze(1).detach().clone()
        #s_jacob = tdists.normal.Normal(s_preds, 1).log_prob(sum_gamma)
        # [Step3] Evaluate log_prob
        log_prob = eval_log_prob(p_type, gamma, y, base_dist)
        nll = -torch.mean(log_prob + sldj + logpsz)

        # [Step4] Generates W-adv example
        
        if w_adv == True:
            w_img = wasserstein_example(x.detach().clone(), y.detach().clone(),
                                        encoder = encoder, lamb = 1.5, max_lr0 = 0.01,
                                        flow = flow, log_type = p_type, flow_type = flow_type,
                                        base_dist = base_dist, s_preds = s_preds)


            if torch.isnan(w_img).any() == True:
                print("[DEBUG] NaN value is detected in W_img")
                sys.exit()

            with torch.no_grad():
                w_latents = encoder(w_img)
            # [Step5] Increase the entropy of W-adv

            w_gamma, w_sldj = flow(w_latents)
            w_gamma /= torch.sum(w_gamma, dim = 1, keepdim = True)
            entropy = torch.sum(w_gamma * torch.log(w_gamma + gamma_eps), dim = 1)
            entropy = torch.mean(entropy)
            
            loss = nll + 6.0*entropy
            #loss = nll
        else:
            loss = nll

        flow.zero_grad()
        #optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(flow.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        if (step+1) % 500 == 0 or step == 1497:
            for param in optimizer.param_groups:
                print("Learning Rate: ", param['lr'])
            cnt = 0
            adv_cnt = 0
            total = 0
            probs = []
            labels = []
            valid_probs = []
            ood_probs = []
            
            for val_batch in val_loader:
                x = val_batch[0].to(device)
                x.requires_grad = True
                y = val_batch[1].to(device)

                with torch.no_grad():
                    latents = encoder(x)
         
                gamma, sldj = flow(latents)
                preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)

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
            np.save("./data/probs/{}_valid_probs_{}.npy".format(dro, s_cnt), probs)
            
            for i, batch in enumerate(train_c_loader):
                cnt, total = (0, 0)
                probs = []
                labels = []
                x = torch.as_tensor(batch[0], device = device, dtype = torch.float32)
                y = torch.as_tensor(batch[1], device = device, dtype = torch.int64)
                
                with torch.no_grad():
                    latents = encoder(x)

                gamma, sldj = flow(latents)

                preds = prediction(gamma, sldj, alphas, p_type, base_dists = base_dist, pi = pi)
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
            '''
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
            np.save('./data/probs/{}_svhn_probs_{}.npy'.format(dro, s_cnt), ood_probs.cpu().detach().numpy())
            np.save('./data/probs/{}_lsun_probs_{}.npy'.format(dro, s_cnt), lsun_probs.cpu().detach().numpy())
            np.save('./data/probs/{}_tiny_probs_{}.npy'.format(dro, s_cnt), tiny_probs.cpu().detach().numpy())
            s_cnt += 1
            '''
        #print("entropy mean: ", torch.mean(entropies))
            #print("entropy[:10]: ", entropies[:10])
#torch.save(flow.state_dict(), os.path.join(flow_path, "adv_b1024_init_e100_L12_epoch{}".format(epoch)))
