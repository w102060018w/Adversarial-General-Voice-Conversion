#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:38:24 2017

@author: chadyang
"""

import torch 
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch.optim as optim
from util import Tanhize, convert_f0, pw2wav
import os
import soundfile as sf
from architecture import GaussianKLD, GaussianLogDensity
import architecture_ctxwindow


#%% parameters
OUTWAVROOT = '/home/chadyang/CEDL/final/Scripts/output/ctxwindow/'
ALLITER = 30000
VAEITER = 10000
#ALLITER = 1
#VAEITER = 1
BATCHSIZE = 256
LAMBDAS = [ 1, 10]
ALPHAS = [ 1, 10, 50]
HIDDIM = 64
SRC = 'SF1'
TRG = 'TM3'

#LAMBDAS = [1]
#ALPHAS=[50]

PLOT = True
GENERATE = True
FS = 16000


#%% funcs     
def generate_random_sample(data, norm=None):
    while True:
        random_indexs = np.random.choice(data.__len__(), size=BATCHSIZE ,replace = False)
        batch = []
        for random_index in random_indexs:
            
            if random_index < 1:
                random_index = 1
            elif random_index >= BATCHSIZE-1:
                random_index = BATCHSIZE-2
                
            data_ctx = normalizer.forward_process(data[random_index])
            for i in [-1,1]:
                #data_ctx.append(data[random_index+i])
                data_ctx = np.hstack((data_ctx,normalizer.forward_process(data[random_index+i])))
                
            batch.append(data_ctx)
            
        #batch = [data[i]for i in random_indexs]
        batch = np.array(batch)
        #print('bt-size = ',batch.shape)
        #if norm != None:
        #    batch = normalizer.forward_process(batch)
        yield torch.from_numpy(batch).float()


#%%  train
for ALPHA in ALPHAS:
    for LAMBDA in LAMBDAS:
        # architecture
        netE, netG, netD = architecture_ctxwindow.vaewgan_dnn_v1(HIDDIM)
        sampler = architecture_ctxwindow.Sampler()
        
        #backward parameter
        one = torch.FloatTensor([1]).cuda()
        mone = one * -1
        
        # setup optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerE = optim.Adam(netE.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        
        #%% load data
        data = joblib.load("/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainFea-2people.pkl")
        s_data = data[:20363]
        t_data = data[20363:]
        
        sid = joblib.load("/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainLabel-2people-2.pkl")
        s_sid = sid[:20363][:BATCHSIZE]
        t_sid = sid[20363:][:BATCHSIZE]
        s_sid = torch.from_numpy(s_sid).float()
        t_sid = torch.from_numpy(t_sid).float()

        normalizer = Tanhize(
            xmax=np.fromfile('./etc/xmax.npf'),
            xmin=np.fromfile('./etc/xmin.npf'),
        )

        loss_D = []
        loss_KLD = []
        loss_logp = []
                
        
        #%% train
        for iter in range(ALLITER):
            
            #==============================================================================
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #==============================================================================
            
            real_data_t = generate_random_sample(t_data, norm=normalizer).__next__()
            real_data_t_v = Variable(real_data_t).cuda()
            real_id_t_v = Variable(t_sid).cuda()
        
            # train with real
            D_real = netD(real_data_t_v)
            D_real = D_real.mean()
        #            D_real.backward(mone)
        
            # train with fake   
            # vae_fake = Variable(netG(real_data_v,real_id_v).data)
            real_data_s = generate_random_sample(s_data, norm=normalizer).__next__()
            real_data_s_v = Variable(real_data_s).cuda()
            
            vae_fake_s2t = netG(sampler(netE(real_data_s_v)), real_id_t_v)
            D_fake = netD(vae_fake_s2t)
            D_fake = D_fake.mean()
        #            D_fake.backward(one)
          
            # gradient peanalty
            alpha = torch.rand(BATCHSIZE, 1)
            alpha = alpha.expand(real_data_t_v.size())
            alpha = alpha.cuda()
        
            interpolates = alpha * real_data_t_v.data + ((1 - alpha) * vae_fake_s2t.data)
            interpolates = torch.autograd.Variable(interpolates, requires_grad=True)    
            disc_interpolates = netD(interpolates)    
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        #            gradient_penalty.backward()
            
            D_cost =  D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            loss_D.append(Wasserstein_D.data[0])      
        
        
        
            #==============================================================================
            # (2) Update G network: VAE
            #==============================================================================
            real_data_t = generate_random_sample(t_data, norm=normalizer).__next__()
            real_data_t_v =Variable(real_data_t).cuda()
            real_id_t_v =Variable(t_sid).cuda()
                
            real_data_s = generate_random_sample(s_data, norm=normalizer).__next__()
            real_data_s_v =Variable(real_data_s).cuda()
            real_id_s_v =Variable(s_sid).cuda()
            
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
            # MSEerr_s = MSECriterion(rec_s, real_data_s_v)
            log_var_s = Variable(torch.FloatTensor(np.zeros(real_data_s_v.data.shape)).cuda())
            logp_s = GaussianLogDensity(real_data_s_v, rec_s, log_var_s)
        
            rec_t = netG(sampler(netE(real_data_t_v)), real_id_t_v)
            #MSEerr_t = MSECriterion(rec_t, real_data_t_v)
            log_var_t = Variable(torch.FloatTensor(np.zeros(rec_s.data.shape)).cuda())
            logp_t = GaussianLogDensity(real_data_t_v, rec_t, log_var_t)
        
            #MSEerr = (MSEerr_s +  MSEerr_t) / 2.0
            MSEerr = (logp_s + logp_t) / -2.0
            loss_logp.append(MSEerr.data[0])
        
            if iter%500==0:
                print(iter, ' | W:{} | KLD:{} | MSE:{}'.format(round((Wasserstein_D).data[0],3), round(KLD.data[0],3), round(MSEerr.data[0],3)))
                
            if iter<VAEITER:
                if iter<5000:
                    netE.zero_grad() 
                    VAEerr = 0*KLD + MSEerr
                    VAEerr.backward(retain_graph=True)
                    optimizerE.step()
                    
                    netG.zero_grad()
                    G_loss = 0*(- D_cost ) + MSEerr
                    G_loss.backward()
                    optimizerG.step()
                else:
                    netE.zero_grad() 
                    VAEerr = KLD + MSEerr
                    VAEerr.backward(retain_graph=True)
                    optimizerE.step()
            
                    # Update G
                    netG.zero_grad()
                    G_loss = 0*(- D_cost ) + MSEerr
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
                    netE.zero_grad() 
                    VAEerr = KLD + MSEerr
                    VAEerr.backward(retain_graph=True)
                    optimizerE.step()
                
                    # Update G
                    netG.zero_grad()
                    G_loss = ALPHA*(- D_cost ) + MSEerr
                    G_loss.backward()
                    optimizerG.step()
        
        
        #%% Save model
        savePath = '/home/chadyang/CEDL/final/Scripts/model/vaegan-dnn{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format( 'v2.3' ,SRC, TRG, ALLITER, VAEITER, BATCHSIZE, HIDDIM, LAMBDA, ALPHA )
        netParam = {'Encoder':netE.state_dict(), 'Generator':netG.state_dict(), 'Discriminator':netD.state_dict()}
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
            plt.show()
            pltPath = '/home/chadyang/CEDL/final/Scripts/model/vaegan-dnn{}-{}-{}-{}-{}-{}-{}-{}-{}.png'.format('v2.3',SRC, TRG, ALLITER, VAEITER, BATCHSIZE, HIDDIM, LAMBDA, ALPHA)
            plt.savefig(pltPath, dpi=300, format="png")
            

        #%% generate
        if GENERATE:
            srgdata = joblib.load('/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/vcc2016_training/'+SRC+'/100002.pkl')
            f0_s = srgdata[:,1026]
            ap_s = srgdata[:, 513:1026]
            en_s = srgdata[:, 1027]
            sp_s = srgdata[:,:513]
            
            sp_s_new = []
            for idx, s in enumerate(sp_s):
                #print(s.shape)
                if idx == 0:
                    norm_re = normalizer.forward_process(sp_s[:3,:])
                    sp_s_new.append(norm_re.reshape(1,-1))
                elif idx == len(sp_s)-1:
                    #print(sp_s[:,-3:].shape)
                    norm_re = normalizer.forward_process(sp_s[-3:,:])
                    sp_s_new.append(norm_re.reshape(1,-1))
                else:
                    norm_re = normalizer.forward_process(sp_s[idx-1:idx+2,:])
                    sp_s_new.append(norm_re.reshape(1,-1))
                
                print(len(sp_s_new[idx][0]))

            sp_s_new = np.concatenate(sp_s_new)
            #print(sp_s_new.shape)
            sp_s_norm = sp_s_new
            #sp_s_norm = normalizer.forward_process(sp_s_new)
            lensp = srgdata.shape[0]
            id_t = np.zeros([lensp, 2])
            id_t[:,1] = 1
            
            sp_s2t_norm = netG(sampler(netE(Variable(torch.from_numpy(sp_s_norm)).cuda())), Variable(torch.from_numpy(id_t).float()).cuda()).data.cpu().numpy()


            sp_t_new = []
            for idx, s in enumerate(sp_s2t_norm):
                #print(s.shape)
                if idx == 0:
                    norm_re = normalizer.backward_process(sp_s2t_norm[0,:513])
                elif idx == len(sp_s)-1:
                    #print(sp_s[:,-3:].shape)
                    norm_re = normalizer.backward_process(sp_s2t_norm[idx,513:513*2])
                else:
                    norm_re = normalizer.backward_process(sp_s2t_norm[idx,-513:])

                sp_t_new.append(norm_re.reshape(1,-1))
             
            sp_s2t = np.concatenate(sp_t_new)
            #print(sp_s2t.shape)
            #sp_s2t = normalizer.backward_process(sp_s2t_norm)
            f0_s2t = convert_f0(f0_s, SRC, TRG)
            
            y_s2t = pw2wav({'sp':sp_s2t, 'ap':ap_s, 'en':en_s, 'f0':f0_s2t})
            oFilename = os.path.join(OUTWAVROOT, '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.wav'.format('v2.3',SRC, TRG, 100002, ALLITER, VAEITER, BATCHSIZE, HIDDIM, LAMBDA, ALPHA))
            wav = sf.write(oFilename, y_s2t, FS)

            
            y_s = pw2wav({'sp':sp_s, 'ap':ap_s, 'en':en_s, 'f0':f0_s})
            oFilename = os.path.join(OUTWAVROOT, '{}-{}-{}.wav'.format(SRC, SRC, 100002))
            wav = sf.write(oFilename, y_s, FS)
#            


#            