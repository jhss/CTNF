import math
import types
import sys

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gamma import Gamma

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        #print("[DEBUG] linear weight type: ", self.linear.weight.data)
        #print("[DEBUG] outputs: ", output)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output

nn.MaskedLinear = MaskedLinear

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]
        #act_func = myact

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            
            if torch.isnan(torch.exp(-a)).any() == True:
                print("[DEBUG] h: ", h)
                print("[DEBUG] NaN value is detected in torch.exp(-a)")
                sys.exit()
            if torch.isnan(m).any() == True:
                print("[DEBUG] NaN value is detected in m")
                sys.exit()
            if torch.isnan(inputs).any() == True:
                print("[DEBUG] NaN value is detected in inputs")
                sys.exit()
            u = (inputs - m) * torch.exp(-a)
            if torch.isnan(u).any() == True:
                print("[DEBUG] NaN value is detected in MADE module")
                sys.exit()
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)

class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                if torch.isinf(inputs).any() == True:
                    print("[DEBUG] inf value is detected in inputs")
                    print(inputs)
                    sys.exit()
                if torch.isinf(self.batch_mean).any() == True:
                    print("[DEBUG] inf value is detected in self.batch_mean")
                    print(self.batch_mean)
                    sys.exit()
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps
                if torch.isinf(self.batch_var).any() == True:
                    print("[DEBUG] inf value is detected in self.batch_var")
                    print("inputs: ", inputs)
                    print("batch_mean: ", self.batch_mean)
                    print("(inputs-self.batch_mean).pow(2): ", (inputs - self.batch_mean).pow(2))
                    print(self.batch_var)
                    sys.exit()

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var
            
            if torch.isnan(inputs).any() == True:
                print("[DEBUG] NaN value is detected in inputs")
                sys.exit()
            if torch.isnan(mean).any() == True:
                print("[DEBUG] NaN value is detected in mean")
                sys.exit()
            if torch.isnan(var.sqrt()).any() == True:
                print("var: ", var)
                print("[DEBUG] NaN value is detected in var.sqrt()")
                sys.exit()

            if torch.isinf(inputs).any() == True:
                print("[DEBUG] inf value is detected in inputs")
                sys.exit()
            if torch.isinf(mean).any() == True:
                print("[DEBUG] inf value is detected in mean")
                sys.exit()
            if torch.isinf(var.sqrt()).any() == True:
                print("var: ", var)
                print("[DEBUG] inf value is detected in var.sqrt()")
                sys.exit()

            x_hat = (inputs - mean) / var.sqrt()
            if torch.isnan(x_hat).any() == True:
                print("[DEBUG] NaN value is detected in x_hat")
                sys.exit()
            if torch.isinf(x_hat).any() == True:
                print("[DEBUG] inf value is detected in x_hat")
                sys.exit()

            y = torch.exp(self.log_gamma) * x_hat + self.beta
            if torch.isnan(y).any() == True:
                print("[DEBUG] Nan Value is detected in BatchNorm Module")
                sys.exit()
            if torch.isinf(y).any() == True:
                print("[DEBUG] inf Value is detected in BatchNorm Module")
                sys.exit()
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)

class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)

class SurNorm(nn.Module):
    """ Surjective normalization layer
    """
    def __init__(self, num_inputs):
        super(SurNorm, self).__init__()
        self.num_inputs = num_inputs
        self.pi = nn.Sequential(nn.Linear(10, 1), nn.ReLU())

    # assume that inputs are detached from a graph
    def forward(self, inputs):
        s = torch.sum(inputs, dim = 1)
        dirichlet = inputs / s.reshape(-1, 1)
        sum_preds = self.pi(dirichlet).squeeze(1)
        #print("sum_preds: ", sum_preds)
        #print("s : ", s)
        logpsz    = Gamma(sum_preds, 1).log_prob(s) - self.num_inputs * torch.log(sum_preds + 10e-9)
        
        #print("Dirichlet: ", dirichlet)
        #print("logpsz: ", logpsz)
        return dirichlet, logpsz.unsqueeze(1)

    def inverse(self, inputs):
        return self.pi(inputs)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """
    def __init__(self, modules, log_type):
        super().__init__(*modules)
        self.log_type = log_type

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        
        if self.log_type == 'dirichlet':
            idx = (inputs < 0)
            sldj_temp = torch.zeros_like(inputs)
            sldj_temp[idx] += inputs[idx]
            if torch.isinf(sldj_temp[idx]).any() == True:
                print("[DEBUG] inf value is detected sldj_temp")
                sys.exit()
            inputs[idx] = torch.exp(inputs[idx])
            inputs[~idx] = torch.log(inputs[~idx] + 1) + 1
            if torch.isinf(inputs[~idx]).any() == True:
                print("[DEBUG] inf value is detected in inputs[~idx]")
                sys.exit()
            sldj_temp[~idx] += -inputs[~idx] + 1
            sldj_temp = torch.sum(sldj_temp, dim = 1)
            logdets += sldj_temp.unsqueeze(1)
        elif self.log_type == 'temp_dirichlet':
            for i in range(10):
                # v term
                inputs[:,i] = torch.sigmoid(inputs[:,i] - 
                                            torch.log(torch.tensor(10. + 1. - (i + 1))))
                logdets += torch.log(inputs[:,i]*(1 - inputs[:,i])).unsqueeze(1)
                
                # s term
                coeff = torch.ones(32).cuda()
                for j in range(0, i):
                    coeff -= inputs[:,j]
                inputs[:,i] = inputs[:,i] * coeff

                logdets += torch.log(coeff).unsqueeze(1)
                        


        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples
