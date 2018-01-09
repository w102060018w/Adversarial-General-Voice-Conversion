#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:38:51 2017

@author: chadyang
"""
import torch 
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import pi


#%% funcs
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
class Sampler(nn.Module):          #reparameterize
    def __init__(self):
        super(Sampler, self).__init__()
        
    def forward(self,input):
        mu = input[0]
        logvar = input[1]
        # half= torch.FloatTensor([0.5]).cuda()
        std = logvar.mul(0.5).exp_() #calculate the STDEV

        eps = torch.FloatTensor(std.size()).normal_() #random normalized noise
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)    


def GaussianLogDensity(x, mu, log_var):
    EPSILON = 1e-6
    # log_var = Variable(torch.FloatTensor(np.zeros(x.data.shape)).cuda())
    c = np.asscalar(np.log(2.*pi))
    var = log_var.exp()
    x_mu2 = (x - mu).pow(2)   # [Issue] not sure the dim works or not?
    x_mu2_over_var = torch.div(x_mu2, var+EPSILON)
    log_prob = -0.5*(c + log_var + x_mu2_over_var)
    log_prob = torch.sum(log_prob, dim=1)
    log_prob = log_prob.mean()   # keep_dims=True,
    return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    EPSILON = 1e-6
    v1 = lv1.exp()
    v2 = lv2.exp()
    mu_diff_sq = (mu1 - mu2).pow(2)
    dimwise_kld = .5 * (
        (lv2 - lv1) + torch.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
    dimwise_kld = torch.sum(dimwise_kld, dim=1)
    kld = dimwise_kld.mean()
    return kld


#%% vaewgan_dnn_v1
def vaewgan_dnn_v1(HIDDIM = 64):
    class _netE(nn.Module):
        def __init__(self):
            super(_netE, self).__init__()
            self.fc1_1 = nn.Linear(513, 256)
            self.fc1_1_bn = nn.BatchNorm1d(256) #1st fully connect and batch normal
            self.fc1_2 = nn.Linear(256, 128)
            self.fc1_2_bn = nn.BatchNorm1d(128) #2nd fully connect and batch normal
            self.fc1_3 = nn.Linear(128, 128)
            self.fc1_3_bn = nn.BatchNorm1d(128)
            self.fc_mean = nn.Linear(128, HIDDIM) # mean for vae
            self.fc_var = nn.Linear(128, HIDDIM) # var for vae
        
        def forward(self,input):
            x = F.leaky_relu(self.fc1_1_bn(self.fc1_1(input)))
            x = F.leaky_relu(self.fc1_2_bn(self.fc1_2(x)))
            x = F.leaky_relu(self.fc1_3_bn(self.fc1_3(x)))
            return [self.fc_mean(x),self.fc_var(x)] #return (sec 1501 by 10)
            
        
    class _netG(nn.Module):
        def __init__(self):
            super(_netG, self).__init__()
            self.fc1_1 = nn.Linear(HIDDIM+2, 128)
            self.fc1_1_bn = nn.BatchNorm1d(128) #1st fully connect and batch normal        
            self.fc1_2 = nn.Linear(128, 256)
            self.fc1_2_bn = nn.BatchNorm1d(256) #2nd fully connect and batch normal
            self.fc1_3 = nn.Linear(256, 256)
            self.fc1_3_bn = nn.BatchNorm1d(256)
            self.fc1_4 = nn.Linear(256, 513)
    
        def forward(self,input,id):
            output = torch.cat([input,id],1)
            output = F.leaky_relu(self.fc1_1_bn(self.fc1_1(output)))
            output = F.leaky_relu(self.fc1_2_bn(self.fc1_2(output)))
            output = F.leaky_relu(self.fc1_3_bn(self.fc1_3(output)))
            output = F.tanh(self.fc1_4(output))
            return output

        
    class _netD(nn.Module):    
        def __init__(self):
            super(_netD, self).__init__()
            self.fc1_1 = nn.Linear(513, 256)
            self.fc1_2 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 1)
            
        def forward(self,input):
            output = F.leaky_relu(self.fc1_1(input))
            output = F.leaky_relu(self.fc1_2(output))
            return self.fc2(output)
        
    return _netE().apply(weights_init).cuda(), _netG().apply(weights_init).cuda(), _netD().apply(weights_init).cuda()


#%% 
def vaewgan_dnn_v2(HIDDIM = 64):
    class _netE(nn.Module):
        def __init__(self):
            super(_netE, self).__init__()
            self.fc1_1 = nn.Linear(513, 256)
            self.fc1_1_bn = nn.BatchNorm1d(256) #1st fully connect and batch normal
            self.fc1_2 = nn.Linear(256, 128)
            self.fc1_2_bn = nn.BatchNorm1d(128) #2nd fully connect and batch normal
            self.fc_mean = nn.Linear(128, HIDDIM) # mean for vae
            self.fc_var = nn.Linear(128, HIDDIM) # var for vae
        
        def forward(self,input):
            x = F.leaky_relu(self.fc1_1_bn(self.fc1_1(input)))
            x = F.leaky_relu(self.fc1_2_bn(self.fc1_2(x)))
            
            return [self.fc_mean(x),self.fc_var(x)] #return (sec 1501 by 10)
            
        
    class _netG(nn.Module):
        def __init__(self):
            super(_netG, self).__init__()
            self.fc1_1 = nn.Linear(HIDDIM+2, 128)
            self.fc1_1_bn = nn.BatchNorm1d(128) #1st fully connect and batch normal 
            
            self.fc1_2 = nn.Linear(128, 256)
            self.fc1_2_bn = nn.BatchNorm1d(256)
            
            self.fc1_22 = nn.Linear(256, 256)
            self.fc1_22_bn = nn.BatchNorm1d(256) #2nd fully connect and batch normal
            
            self.fc1_23 = nn.Linear(256, 256)
            self.fc1_23_bn = nn.BatchNorm1d(256)
            self.fc1_3 = nn.Linear(256, 513)
    
        def forward(self,input,id):
            output = torch.cat([input,id],1)
            output = F.leaky_relu(self.fc1_1_bn(self.fc1_1(output)))
            output = F.leaky_relu(self.fc1_2_bn(self.fc1_2(output)))
            output = F.leaky_relu(self.fc1_22_bn(self.fc1_22(output)))
            output = F.leaky_relu(self.fc1_23_bn(self.fc1_23(output)))
            output = self.fc1_3(output)
            return output

        
    class _netD(nn.Module):    
        def __init__(self):
            super(_netD, self).__init__()
            self.fc1_1 = nn.Linear(513, 256)
            self.fc1_2 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 1)
            
        def forward(self,input):
            output = F.leaky_relu(self.fc1_1(input))
            output = F.leaky_relu(self.fc1_2(output))
            return self.fc2(output)
        
    return _netE().apply(weights_init).cuda(), _netG().apply(weights_init).cuda(), _netD().apply(weights_init).cuda()

#v3
def vaewgan_dnn_v3(HIDDIM = 64):
    class _netE(nn.Module):
        def __init__(self):
            super(_netE, self).__init__()
            self.fc1_1 = nn.Linear(1026, 512)
            self.fc1_1_bn = nn.BatchNorm1d(512) #1st fully connect and batch normal
            self.fc1_2 = nn.Linear(512, 256)
            self.fc1_2_bn = nn.BatchNorm1d(256) #2nd fully connect and batch normal
            self.fc_mean = nn.Linear(256, HIDDIM) # mean for vae
            self.fc_var = nn.Linear(256, HIDDIM) # var for vae
        
        def forward(self,input):
            x = F.leaky_relu(self.fc1_1_bn(self.fc1_1(input)))
            x = F.leaky_relu(self.fc1_2_bn(self.fc1_2(x)))
            
            return [self.fc_mean(x),self.fc_var(x)] #return (sec 1501 by 10)
            
        
    class _netG(nn.Module):
        def __init__(self):
            super(_netG, self).__init__()
            self.fc1_1 = nn.Linear(HIDDIM+2, 256)
            self.fc1_1_bn = nn.BatchNorm1d(256) #1st fully connect and batch normal 
            
            self.fc1_2 = nn.Linear(256, 256)
            self.fc1_2_bn = nn.BatchNorm1d(256)
            
            self.fc1_22 = nn.Linear(256, 512)
            self.fc1_22_bn = nn.BatchNorm1d(512) #2nd fully connect and batch normal
            
            self.fc1_23 = nn.Linear(512, 512)
            self.fc1_23_bn = nn.BatchNorm1d(512)
            self.fc1_3 = nn.Linear(512, 1026)
    
        def forward(self,input,id):
            output = torch.cat([input,id],1)
            output = F.leaky_relu(self.fc1_1_bn(self.fc1_1(output)))
            output = F.leaky_relu(self.fc1_2_bn(self.fc1_2(output)))
            output = F.leaky_relu(self.fc1_22_bn(self.fc1_22(output)))
            output = F.leaky_relu(self.fc1_23_bn(self.fc1_23(output)))
            output = self.fc1_3(output)
            return output

        
    class _netD(nn.Module):    
        def __init__(self):
            super(_netD, self).__init__()
            self.fc1_1 = nn.Linear(1026, 512)
            self.fc1_2 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 1)
            
        def forward(self,input):
            output = F.leaky_relu(self.fc1_1(input))
            output = F.leaky_relu(self.fc1_2(output))
            return self.fc2(output)
        
    return _netE().apply(weights_init).cuda(), _netG().apply(weights_init).cuda(), _netD().apply(weights_init).cuda()


#%% lstm
def vaewgan_lstm_v1(TIMESTEP=8, HIDDIM=64):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.0)
            m.bias.data.fill_(0)    

    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
        def forward(self, input):
            mean = input[0]
            lvar = input[1]
            norm = Variable(torch.randn(lvar.size())).cuda()
            return mean + (lvar / 2).exp() * norm
    
    class _netE(nn.Module):
        def __init__(self):
            super(_netE, self).__init__()
            self.ls_0 = nn.LSTM(513, 256, batch_first = True)
            self.ls_1 = nn.LSTM(256, 128, batch_first = True)
            self.mean = nn.Linear(128, HIDDIM)
            self.lvar = nn.Linear(128, HIDDIM)
        def forward(self, input):
            w0 = Variable(torch.randn(1, input.size(0), 256)).cuda()
            w1 = Variable(torch.randn(1, input.size(0), 128)).cuda()
            s0 = Variable(torch.zeros(1, input.size(0), 256)).cuda()
            s1 = Variable(torch.zeros(1, input.size(0), 128)).cuda()
            h, _ = self.ls_0(input, (w0, s0))
            h, _ = self.ls_1(h, (w1, s1))
            mean = self.mean(h[:,-1,:])
            lvar = self.lvar(h[:,-1,:])
            return mean, lvar # mean, lvar
    
    class _netG(nn.Module):
        def __init__(self):
            super(_netG, self).__init__()
            self.li_0 = nn.Linear(HIDDIM+2, 128)
            self.ls_0 = nn.LSTM(128, 256, batch_first = True)
            self.ls_1 = nn.LSTM(256, 513, batch_first = True)
        def forward(self, input, sid):
            x = self.li_0(torch.cat([input, sid], 2))
            
            
            input = input.unsqueeze(1)
            input = input.expand(input.size(0), TIMESTEP, input.size(2))
            sid = sid.unsqueeze(1)
            sid = sid.expand(sid.size(0), TIMESTEP, sid.size(2))
            w0 = Variable(torch.randn(1, input.size(0), 128)).cuda()
            w1 = Variable(torch.randn(1, input.size(0), 256)).cuda()
            s0 = Variable(torch.zeros(1, input.size(0), 128)).cuda()
            s1 = Variable(torch.zeros(1, input.size(0), 256)).cuda()
            # x = 
            h, _  = self.ls_0(x, (w0, s0))
            h, _  = self.ls_1(h, (w1, s1))
            # y = self.li_0(h)
            return h
    
    class _netD(nn.Module):
        def __init__(self):
            super(_netD, self).__init__()
            self.ls_0 = nn.LSTM(513, 128, batch_first = True)
            self.li_0 = nn.Linear(128, 10)
            self.li_1 = nn.Linear(10, 1)
        def forward(self, input):
            w0 = Variable(torch.randn(1, input.size(0), 128)).cuda()
            s0 = Variable(torch.zeros(1, input.size(0), 128)).cuda()
            h, _  = self.ls_0(input, (w0, s0))
            output = self.li_0(h[:, -1, :])
            output = self.li_1(output)
            return output
            
    return _netE().apply(weights_init).cuda(), _netG().apply(weights_init).cuda(), _netD().apply(weights_init).cuda(), Sampler().cuda()



#%% CNN
def cnn(HIDDIM=128):
    class _netE(nn.Module):
        def __init__(self):
            super(_netE, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),  # b, 16, 85, 85
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 16, 3, stride=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2)  # b, 16, 14, 14
            )
            self.mean = nn.Linear(16*14*14, HIDDIM)
            self.lvar = nn.Linear(16*14*14, HIDDIM)

        def forward(self, input):
            x = self.conv1(input)
            x = self.conv2(x)
            mean = self.mean(x.view(x.size()[0], -1))
            lvar = selfx.lvar(x.view(x.size()[0], -1))
            return mean, lvar
            

    class Sampler(nn.Module): #reparameterize
        def __init__(self):
            super(Sampler, self).__init__()
            
        def forward(self,input):
            mu = input[0]
            logvar = input[1]
            # half= torch.FloatTensor([0.5]).cuda()
            std = logvar.mul(0.5).exp_() #calculate the STDEV
    
            eps = torch.FloatTensor(std.size()).normal_() #random normalized noise
            eps = Variable(eps).cuda()
            return eps.mul(std).add_(mu)    

#            self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
#            nn.Conv2d(
#                in_channels=1,      # input height
#                out_channels=16,    # n_filters
#                kernel_size=5,      # filter size
#                stride=1,           # filter movement/step
#                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
#            ),      # output shape (16, 28, 28)
#            nn.ReLU(),    # activation
#            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
#            )
#            self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
#            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
#            nn.ReLU(),  # activation
#            nn.MaxPool2d(2),  # output shape (32, 7, 7)
#            )
#            self.out = nn.Linear(32*7*7, 10)   # fully connected layer, output 10 classes

#            self.fc1_1 = nn.Linear(1026, 512)
#            self.fc1_1_bn = nn.BatchNorm1d(512) #1st fully connect and batch normal
#            self.fc1_2 = nn.Linear(512, 256)
#            self.fc1_2_bn = nn.BatchNorm1d(256) #2nd fully connect and batch normal
#            self.fc_mean = nn.Linear(256, HIDDIM) # mean for vae
#            self.fc_var = nn.Linear(256, HIDDIM) # var for vae


    class _netG(nn.Module):
        def __init__(self):
            super(_netG, self).__init__()
            self.fc1 = nn.Linear(HIDDIM+2, 16*14*14)

            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(16, 16, 3, stride=3, padding=1),  # b, 16, 5, 5
                nn.ReLU(True)
            )

            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(16, 16, 3, stride=3, padding=1),  # b, 16, 5, 5
                nn.Tanh()
            )

        def forward(self, input, sid):
            output = torch.cat([input,id],1)
            output = self.fc1(output)
            output = output.view(-1, 16, 14, 14)
            
            output = output.unsqueeze(1)
            output = output.unsqueeze(1)
            
            return h


    
    class D:




