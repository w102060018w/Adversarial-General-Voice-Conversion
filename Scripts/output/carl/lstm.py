#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 19:44:57 2017

@author: chadyang
"""


import torch 
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
#import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi

from architecture import GaussianKLD, GaussianLogDensity
import architecture

#%% parameters
ALLITER = 10000
VAEITER = 10000
BATCHSIZE = 64
LAMBDAS = [1]
ALPHAS = [0]


#LAMBDA = 10
#ALPHA = 10
TYPE = 'lstm_dense_PURE_VAE'
PLOT = True
GENERATE = True
TIMESTEPS = [1000]
LSTM_HIDDIMS = [128]
TAKE_EVERY_TIME_STEP_OUTPUTS = [True]


#TAKE_EVERY_TIME_STEP_OUTPUT = False
#TIMESTEP = 8
#LSTM_HIDDIM = 60
#DENSE_DIM = LSTM_HIDDIM//2
#HIDDIM = LSTM_HIDDIM//4

#%% model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
    dimwise_kld = .5 * ((lv2 - lv1) + torch.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
    dimwise_kld = torch.sum(dimwise_kld, dim=1)
    kld = dimwise_kld.mean()
    return kld


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


class _netE(nn.Module): #encoder
    def __init__(self):
        super(_netE, self).__init__()
        ## dnn : 64*513 (bt-size * fea-dim)
        ## rnn : (64*5)*513 (bt-size * time-step * input-size)
        self.rnn_num_layers = 1
        self.rnn_hidden_size = LSTM_HIDDIM # set the hidden size as u want
        
        self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=513,
            hidden_size=self.rnn_hidden_size,         # rnn hidden unit
            num_layers=self.rnn_num_layers,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, TIMESTEP, input_size)
        )

        self.dense = nn.Sequential(
            nn.Linear(LSTM_HIDDIM, DENSE_DIM), # 256 => 128
            nn.LeakyReLU(),
        )
        self.mean = nn.Linear(DENSE_DIM, HIDDIM)  # 128 -> 64
        self.var = nn.Linear(DENSE_DIM, HIDDIM)  # 128 -> 64
    
    def forward(self,input):
        # Set initial states 
        h0 = Variable(torch.zeros(self.rnn_num_layers, input.size(0), self.rnn_hidden_size)).cuda() 
        c0 = Variable(torch.zeros(self.rnn_num_layers, input.size(0), self.rnn_hidden_size)).cuda()
        
        # Forward propagate RNN
        out, _ = self.lstm(input, (h0, c0)) # out: 64x8x256
        
        # Decode hidden state of last time step and go through encoder
        if TAKE_EVERY_TIME_STEP_OUTPUT == True:
            z = self.dense(out) # z: 64*8*128
        else:
            z = self.dense(out[:, -1, :]) # 64*128

	#z = out[:, -1, :] # 64*128 
        return [self.mean(z), self.var(z)] # mu:64*8*HIDDIM, sd:64*8*HIDDIM OR # mu:64*HIDDIM, ds:64*HIDDIM

        
    
class _netG(nn.Module): # generator
    def __init__(self):
        super(_netG, self).__init__()
        # fc -> rnn
        # -(dnn) 66->128->256 -(rnn) 64*8*256 -> 64*8*513
        
        self.rnn_num_layers = 1
        self.rnn_hidden_size = 513 # set the hidden size as u want
        
        self.dense = nn.Sequential(
            nn.Linear(HIDDIM+2, DENSE_DIM),
            nn.LeakyReLU(),
            nn.Linear(DENSE_DIM, DENSE_DIM*2), 
        )
        
        self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            #input_size = HIDDIM+2,
            input_size = DENSE_DIM*2,
            hidden_size = self.rnn_hidden_size,         # rnn hidden unit
            num_layers = self.rnn_num_layers,           # number of rnn layer
            batch_first = True,       # input & output will has batch size as 1s dimension. e.g. (batch, TIMESTEP, input_size)
        )
        
#        self.fc = nn.Linear(self.rnn_hidden_size, 513)

        

    def forward(self,input,sid):
#         h0 = Variable(torch.zeros(self.rnn_num_layers, input.size(0), self.rnn_hidden_size)).cuda() 
#         c0 = Variable(torch.zeros(self.rnn_num_layers, input.size(0), self.rnn_hidden_size)).cuda()
        h0 = Variable(torch.zeros(self.rnn_num_layers, input.size(0), self.rnn_hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.rnn_num_layers, input.size(0), self.rnn_hidden_size)).cuda()
#         print(input) # 64*64 (the output of 'sampler')

        #broad test on using "out" or "out[:,-1,:]" in Encoder, so the dimension in Decoder should also modify in flexible way.
        if TAKE_EVERY_TIME_STEP_OUTPUT == False:
            #print('Double check on Decoder-Output dimension: ',len(input.size()))
            input = input.unsqueeze(1)
            input = input.expand(input.size(0), TIMESTEP, input.size(2)) # duplicate result: 64*8*66
            
        sid = sid.unsqueeze(1)
        sid = sid.expand(sid.size(0), TIMESTEP, sid.size(2))
        
        output = torch.cat([input, sid],2) # 64*8*66
        
	# add dense structure
        output = F.relu(self.dense(output)) # 64*8*256

        # output = output.view(TIMESTEP,-1,output.size()[1]) # 8*8*256
        
        # rnn forward process
        out, _ = self.lstm(output, (h0, c0))  #64*8*513
        # print('decoder lstm output : ',out)
        # print('ori output = ',out)
        # print('test output = ', out[:,-1,:])
#        out = self.fc(out[:, -1, :]) 
        
        return out

#%%  train
for TAKE_EVERY_TIME_STEP_OUTPUT in TAKE_EVERY_TIME_STEP_OUTPUTS:
    for TIMESTEP in TIMESTEPS:
        for LSTM_HIDDIM in LSTM_HIDDIMS:
            for ALPHA in ALPHAS:
                for LAMBDA in LAMBDAS:        

                    print('==================')
                    print('TIMESTEP : ',TIMESTEP)
                    print('LSTM_HIDDIM : ',LSTM_HIDDIM)
                    print('==================')
                    
                    DENSE_DIM = LSTM_HIDDIM//2
                    HIDDIM = LSTM_HIDDIM//4

                    if TAKE_EVERY_TIME_STEP_OUTPUT == True:
                        print('We will take "every time step" output from the LSTM result. (i.e. output-dim from Encoder = bt-size*time-step*128)')
                    else:
                        print('We will take "only the last time step" output from the LSTM result. (i.e. output-dim from Encoder = bt-size*128)')

                    # initialize
                    # D init
                    #netD=_netD()
                    #netD.apply(weights_init).cuda()

                    # E init
                    netE = _netE() # input: s_data: 64*513
                    netE.apply(weights_init).cuda()

                    # G init
                    netG=_netG()
                    netG.apply(weights_init).cuda()

                    sampler = Sampler()
                    sampler.apply(weights_init).cuda()


                    # netE, netG, netD, sampler = architecture.vaewgan_lstm_v1()

                    # mse loss
#                    mse_s = nn.MSELoss().cuda()
#                    mse_t = nn.MSELoss().cuda()


                    #backward parameter
                    one = torch.FloatTensor([1]).cuda()
                    mone = one * -1

                    # setup optimizer
                    #optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
                    optimizerE = optim.Adam(netE.parameters(), lr=0.0001, betas=(0.5, 0.999))
                    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))


                    #%% load data
                    data = joblib.load("/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainFea-2people.pkl")
                    s_data = data[:20363]
                    t_data = data[20363:]

                    sid = joblib.load("/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainLabel-2people-2.pkl")
                    s_sid=sid[:20363][:64]
                    t_sid=sid[20363:][:64]

                    s_sid = torch.from_numpy(s_sid).float()
                    t_sid = torch.from_numpy(t_sid).float()

                    def generate_random_sample(data):
                        while True:
                            random_mat = []
                            for i in range(BATCHSIZE):
                                random_index = np.random.choice(data.__len__(), size=1 ,replace = False)[0]
                                if random_index > data.__len__()-TIMESTEP:
                                    random_index -= TIMESTEP
                                random_ary = list(range(random_index,random_index+TIMESTEP))
                                random_mat.append(data[random_ary,:])
                            random_mat = np.asarray(random_mat)
                            yield torch.from_numpy(random_mat).float()


                    loss_D = []
                    loss_KLD = []
                    loss_logp = []


                    #%% train
                    for iter in range(ALLITER):

                        #==============================================================================
                        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                        #==============================================================================

                        #==============================================================================
                        # (2) Update G network: VAE
                        #==============================================================================
                        real_data_t = generate_random_sample(t_data).__next__()
                        real_data_t_v =Variable(real_data_t).cuda()
                        real_id_t_v =Variable(t_sid).cuda()

                        real_data_s = generate_random_sample(s_data).__next__()
                        real_data_s_v = Variable(real_data_s).cuda()
                        real_id_s_v = Variable(s_sid).cuda()

                        #KL    
                        mu_s , logvar_s = netE(real_data_s_v)
            #            KLD_element_s = mu_s.pow(2).add_(logvar_s.exp()).mul_(-1).add_(1).add_(logvar_s)
            #            KLD_s = torch.sum(KLD_element_s).mul_(-0.5)
                        mu_s_zero = Variable(torch.FloatTensor(np.zeros(mu_s.data.shape))).cuda()
                        logvar_s_zero = Variable(torch.FloatTensor(np.zeros(logvar_s.data.shape))).cuda()
                        KLD_s = GaussianKLD(mu_s, logvar_s, mu_s_zero, logvar_s_zero)

                        mu_t , logvar_t = netE(real_data_t_v)
            #            KLD_element_t = mu_t.pow(2).add_(logvar_t.exp()).mul_(-1).add_(1).add_(logvar_t)
            #            KLD_t = torch.sum(KLD_element_t).mul_(-0.5)
                        mu_t_zero = Variable(torch.FloatTensor(np.zeros(mu_t.data.shape))).cuda()
                        logvar_t_zero = Variable(torch.FloatTensor(np.zeros(logvar_t.data.shape))).cuda()
                        KLD_t = GaussianKLD(mu_t, logvar_t, mu_t_zero, logvar_t_zero)

                        KLD = (KLD_s + KLD_t) / 2.0
                        loss_KLD.append(KLD.data[0])


                        #MSE
                        rec_s = netG(sampler(netE(real_data_s_v)), real_id_s_v)
#                        MSEerr_s = mse_s(rec_s, real_data_s_v)
                        log_var_s = Variable(torch.FloatTensor(np.zeros(real_data_s_v.data.shape)).cuda())
                        logp_s = GaussianLogDensity(real_data_s_v, rec_s, log_var_s)

                        rec_t = netG(sampler(netE(real_data_t_v)), real_id_t_v)
#                        MSEerr_t = mse_t(rec_t, real_data_t_v)
                        log_var_t = Variable(torch.FloatTensor(np.zeros(rec_s.data.shape)).cuda())
                        logp_t = GaussianLogDensity(real_data_t_v, rec_t, log_var_t)

#                        MSEerr = (MSEerr_s +  MSEerr_t) / 2.0
                        MSEerr = (logp_s + logp_t) / -2.0
                        loss_logp.append(MSEerr.data[0])

                        if iter%500==0:
                            #print(iter, ' | W:{} | KLD:{} | MSE:{}'.format(round((Wasserstein_D).data[0],3), round(KLD.data[0],3), round(MSEerr.data[0],3)))
                            print(iter, ' | MSE:{}'.format(round(MSEerr.data[0],3)))

                        if iter<VAEITER:
                            # Update E
                            sampler.zero_grad()
                            netE.zero_grad()
                            #VAEerr = KLD + MSEerr
                            VAEerr = MSEerr
                            VAEerr.backward(retain_graph=True)
                            optimizerE.step()

                            # Update G
                            netG.zero_grad()
                            # G_loss = ALPHA*(-D_cost)+MSEerr
                            G_loss = MSEerr
                            G_loss.backward()
                            optimizerG.step()
                        else:
                            if (iter%5 != 0):
                                # Update D
                                netD.zero_grad() 
                                D_cost.backward()
                                optimizerD.step()
                            else:
                                # Update E
                                sampler.zero_grad()
                                netE.zero_grad()
                                #VAEerr = KLD + MSEerr
                                VAEerr = MSEerr
                                VAEerr.backward(retain_graph=True)
                                optimizerE.step()

                                # Update G
                                netG.zero_grad()
                                G_loss = ALPHA*(-D_cost) + MSEerr
                                G_loss.backward()
                                optimizerG.step()


                    #%% Save model
                    if PLOT:
                        savePath = '/home/chadyang/CEDL/final/Scripts/model/vaegan-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(TYPE, ALLITER, VAEITER, BATCHSIZE, TIMESTEP, HIDDIM, LAMBDA, ALPHA)
                        #netParam = {'Encoder':netE.state_dict(), 'Generator':netG.state_dict(), 'Discriminator':netD.state_dict()}
                        netParam = {'Encoder':netE.state_dict(), 'Generator':netG.state_dict()}
                        joblib.dump(netParam, savePath)


                    #%% plot
                    if PLOT:
                        fig = plt.figure()
                        loss_D = np.array(loss_D)
                        plt.plot(loss_D, label='Discriminator', alpha=0.5)        
                        plt.title("W distance")
                        plt.legend()
                    #            plt.show()

                        loss_KLD = np.array(loss_KLD)
                        plt.plot(loss_KLD, label='KLD', alpha=0.5)        
                        plt.title("VAE KLD")
                        plt.legend()
                    #            plt.show()        

                        loss_logp = np.array(loss_logp)
                        plt.plot(loss_logp, label='log-probability', alpha=0.5)        
                        plt.title("VAE logP")
                        plt.legend()
                    #            plt.show()
                        pltPath = '/home/chadyang/CEDL/final/Scripts/model/vaegan-{}-{}-{}-{}-{}-{}-{}-{}.png'.format(TYPE, ALLITER, VAEITER, BATCHSIZE, TIMESTEP, HIDDIM, LAMBDA, ALPHA)
                        plt.savefig(pltPath, dpi=300, format="png")



                    #%%generate
                    from util import Tanhize, convert_f0, pw2wav
                    import soundfile as sf
                    import os
                    SRC = 'SF1'
                    TRG = 'TM3'
                    FS = 16000
                    OUTWAVROOT = '/home/chadyang/CEDL/final/Scripts/output/carl'
                    normalizer = Tanhize(
                            xmax=np.fromfile('./etc/xmax.npf'),
                            xmin=np.fromfile('./etc/xmin.npf'),
                            )
                    if GENERATE:
                        srgdata = joblib.load('/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/vcc2016_training/'+SRC+'/100002.pkl')
                        f0_s = srgdata[:,1026]
                        ap_s = srgdata[:, 513:1026]
                        en_s = srgdata[:, 1027]
                        sp_s = srgdata[:,:513]

                        padding = TIMESTEP - len(sp_s)%TIMESTEP
                        f0_s = np.concatenate([f0_s, np.zeros([padding])])
                        ap_s = np.concatenate([ap_s, np.zeros([padding, 513])], axis=0)
                        sp_s = np.concatenate([sp_s, np.zeros([padding, 513])], axis=0)
                        en_s = np.concatenate([en_s, np.zeros([padding])])

                        # generate conversion voice s2t
                        sp_s_norm = normalizer.forward_process(sp_s).reshape(-1,TIMESTEP,513)
                        id_t = np.zeros([sp_s_norm.shape[0], 2])
                        id_t[:,1] = 1

                        sp_s2t_norm = netG(sampler(netE(Variable(torch.from_numpy(sp_s_norm)).cuda())), Variable(torch.from_numpy(id_t).float()).cuda()).data.cpu().numpy()
                        sp_s2t_norm = sp_s2t_norm.reshape(-1,513)
                        sp_s2t = normalizer.backward_process(sp_s2t_norm)
                        f0_s2t = convert_f0(f0_s, SRC, TRG)

                        y_s2t = pw2wav({'sp':sp_s2t, 'ap':ap_s, 'en':en_s, 'f0':f0_s2t})
                        oFilename = os.path.join(OUTWAVROOT, '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.wav'.format(TYPE, SRC, TRG, 100002, ALLITER, VAEITER, BATCHSIZE, TIMESTEP, LSTM_HIDDIM, TAKE_EVERY_TIME_STEP_OUTPUT))
                        wav = sf.write(oFilename, y_s2t, FS)

                        # generate reconstruct voice s2s
                        sp_s_norm = normalizer.forward_process(sp_s).reshape(-1,TIMESTEP,513)
                        id_s = np.zeros([sp_s_norm.shape[0], 2])
                        id_s[:,0] = 1

                        sp_s2s_norm = netG(sampler(netE(Variable(torch.from_numpy(sp_s_norm)).cuda())), Variable(torch.from_numpy(id_s).float()).cuda()).data.cpu().numpy()
                        sp_s2s_norm = sp_s2s_norm.reshape(-1,513)
                        sp_s2s = normalizer.backward_process(sp_s2s_norm)

                        y_s2s = pw2wav({'sp':sp_s2s, 'ap':ap_s, 'en':en_s, 'f0':f0_s})
                        oFilename = os.path.join(OUTWAVROOT, '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.wav'.format(TYPE, SRC, SRC, 100002, ALLITER, VAEITER, BATCHSIZE, TIMESTEP, LSTM_HIDDIM, TAKE_EVERY_TIME_STEP_OUTPUT))
                        wav = sf.write(oFilename, y_s2s, FS)




            #            y_s = pw2wav({'sp':sp_s, 'ap':ap_s, 'en':en_s, 'f0':f0_s})
            #            oFilename = os.path.join(OUTWAVROOT, '{}-{}-{}-{}.wav'.format(TYPE, SRC, SRC, 100002))
            #            wav = sf.write(oFilename, y_s, FS)
