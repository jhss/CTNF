import torch
import torch.nn.functional as F
import torch.distributions as tdists
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch.autograd import Variable
import utils
import sys
import numpy as np

def one_hot(y):
    return torch.eye(10)[y]

def inverse_one_hot(y):
    return 1 - torch.eye(10)[y]

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

def prediction(gamma, sldj, alphas, p_type, base_dists, pi):
    
    batch = gamma.shape[0]
    eps = 10e-9

    if p_type == 'gamma':
        base_dist = [[Gamma(alphas[0], 1) if i == j else 
                       Gamma(alphas[1], 1) for i in range(10)] for j in range(10)]    
        batch_categorical_dist = torch.zeros(batch, 10)
        for i in range(batch):
            per_batch_categorical_dist = torch.zeros(1,10)
            for class_index in range(10):
                dists = base_dist[class_index]
                per_class_log_prob = 0
                for j in range(10):
                    temp = dists[j].log_prob(gamma[i,j])
                    per_class_log_prob += temp
                a = sldj[i]
                per_batch_categorical_dist[0,class_index] = per_class_log_prob + sldj[i]
            batch_categorical_dist[i] = per_batch_categorical_dist
        
        batch_categorical_dist = F.softmax(batch_categorical_dist, dim = 1)
    elif p_type == 'gamma_temp':
        gamma /= torch.sum(gamma, dim = 1).reshape(-1, 1)
        batch_categorical_dist = gamma
    elif p_type == 'temp_dirichlet':
        base_dist = []
        for i in range(10):
            concentration = alphas[1] * torch.ones(11).cuda()
            concentration[i] = alphas[0]
            base_dist.append(Dirichlet(concentration))
        batch_categorical_dist = torch.zeros(batch, 10)
        print(gamma.shape)
        sys.exit()
        for i in range(batch):
            for class_index in range(10):
                base_prob = base_dist[class_index].log_prob(gamma[i] + eps)
                batch_categorical_dist[i,class_index] = base_dist[class_index].log_prob(gamma[i] + eps) + sldj[i]

        batch_categorical_dist = stable_softmax(batch_categorical_dist)


    elif p_type == 'dirichlet':
        
        gamma /= torch.sum(gamma, dim = 1).reshape(-1, 1)
        logpsz = eval_logpsz(gamma, pi(gamma).squeeze(1)).unsqueeze(1)
        sldj += logpsz

        if torch.isinf(gamma).any() == True:
            print("[DEBUG] inf value is detected in gamma")
            sys.exit()

        if torch.isinf(sldj).any() == True:
            print("[DEBUG] inf value is detected in sldj")
            sys.exit()

        base_dist = []
        for i in range(10):
            concentration = alphas[1] * torch.ones(10).cuda()
            concentration[i] = alphas[0]
            base_dist.append(Dirichlet(concentration))
        batch_categorical_dist = torch.zeros(batch, 10)

        for i in range(batch):
            for class_index in range(10):
                base_prob = base_dist[class_index].log_prob(gamma[i] + eps)
                if torch.isinf(base_prob).any() == True:
                    print("[DEBUG] inf value is detected in base_prob")
                    print("gamma[i]: ", gamma[i], "class_index ", class_index)
                    sys.exit()
                batch_categorical_dist[i,class_index] = base_dist[class_index].log_prob(gamma[i] + eps) + sldj[i]

        if torch.isinf(batch_categorical_dist).any() == True:
            print("[DEBUG] inf value is detected in batch_categorical_dist")
            sys.exit()

        if torch.isnan(batch_categorical_dist).any() == True:
            print("[DEBUG] NaN value is detected in previous batch_categorical_dist")
            sys.exit()

        batch_categorical_dist = stable_softmax(batch_categorical_dist)

        if torch.isnan(batch_categorical_dist).any() == True:
            print("[DEBUG] NaN value is detected in subsequential batch_categorical_dist")
            sys.exit()

    elif p_type == 'gmm':
        batch_categorical_dist = torch.zeros(batch, 10)
        for i in range(batch):
            for class_index in range(10):
                batch_categorical_dist[i, class_index] = base_dists[class_index].log_prob(gamma[i]) + sldj[i]
        batch_categorical_dist = F.softmax(batch_categorical_dist, dim = 1)
    elif p_type == 'likelihood':
        gamma /= torch.sum(gamma, dim = 1).reshape(-1, 1)
        base_dist = []
        for i in range(10):
            concentration = alphas[1] * torch.ones(10).cuda()
            concentration[i] = alphas[0]
            base_dist.append(Dirichlet(concentration))
        batch_categorical_dist = torch.zeros(batch, 10)
        for i in range(batch):
            for class_index in range(10):
                batch_categorical_dist[i,class_index] = base_dist[class_index].log_prob(gamma[i]) + sldj[i]
        batch_categorical_dist = torch.sum(batch_categorical_dist, dim = 1)

    return batch_categorical_dist

def entropy(preds):
    eps = 10e-9
    return -np.sum(preds * np.log(preds + eps), axis = 1)

def eval_log_prob(log_type, rvs, labels, base_dists):
    
    eps = 10e-9
    log_probs = []

    if log_type == 'gamma':
        rvs += eps
        for i in range(rvs.shape[0]):
            dists = base_dists[labels[i]]
            log_prob = []
            for j in range(10):
                temp = dists[j].log_prob(rvs[i,j]).cuda()
                log_prob.append(temp)
            log_prob = torch.stack(log_prob)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)
        log_probs = torch.sum(log_probs, dim = 1)
    elif log_type == 'dirichlet':
        rvs /= torch.sum(rvs, dim = 1).reshape(-1, 1)
        rvs += eps
        for i in range(rvs.shape[0]):
            log_probs.append(base_dists[labels[i]].log_prob(rvs[i]))
        log_probs = torch.stack(log_probs)
    elif log_type == 'temp_dirichlet':
        rvs_temp = torch.zeros(rvs.shape[0], 11).cuda()
        rvs_temp[:,:10] = rvs
        rvs_temp[:,10] = 1 - torch.sum(rvs_temp[:,:10], dim = 1)
        for i in range(rvs.shape[0]):
            log_probs.append(base_dists[labels[i]].log_prob(rvs_temp[i]))
        log_probs = torch.stack(log_probs)
    elif log_type == 'gmm':
        for i in range(rvs.shape[0]):
            log_probs.append(base_dists[labels[i]].log_prob(rvs[i]))
        log_probs = torch.stack(log_probs)

    return log_probs

def eval_logpsz(gamma, s_preds):
    v = torch.sum(gamma, dim = 1)
    #logpsz    = tdists.normal.Normal(s_preds, 1).log_prob(sum_gamma)
    logpsz = tdists.gamma.Gamma(s_preds, 1).log_prob(v) - 10*torch.log(s_preds + 10e-9)

    return logpsz

def wasserstein_example(x, y, lamb, max_lr0, encoder, flow, log_type, flow_type, base_dist,
                             s_preds):
    
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
        latents = encoder(z_hat)
        gamma, sldj = flow(latents)

        log_prob = eval_log_prob(log_type, gamma, y, base_dist)
        logpsz   = eval_logpsz(gamma, s_preds)
        nll = -torch.mean(log_prob + sldj + logpsz)
        loss = - (nll - lamb * rho)
        loss.backward()
        optimizer_zt.step()

    return z_hat.detach().clone()
