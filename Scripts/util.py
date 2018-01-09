#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:22:09 2017

@author: chadyang
"""
import numpy as np
import joblib
import os
import pyworld as pw


#%% parameters
F0PATH = '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/f0/'


#%% 
class Tanhize(object):
    ''' Normalizing `x` to [-1, 1] '''
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.xscale = xmax - xmin
    
    def forward_process(self, x):
        x = (x - self.xmin) / self.xscale
        return (np.clip(x, 0., 1.) * 2. - 1.).astype('float32')

    def backward_process(self, x):
        return ((x * .5 + .5) * self.xscale + self.xmin).astype('float32')

def convert_f0(f0, src, trg):
    mu_s, std_s = joblib.load(os.path.join(F0PATH, '{}.pkl'.format(src)))
    mu_t, std_t = joblib.load(os.path.join(F0PATH, '{}.pkl'.format(trg)))
    lf0 = np.piecewise(f0, [f0>1., f0<=1.], [lambda x:np.log(x), lambda x:x])    
    lf0 = np.piecewise(lf0, [lf0>1., lf0<=1.], [lambda x:(x-mu_s)/std_s*std_t + mu_t, lambda x:x])    
    lf0 = np.piecewise(lf0, [lf0>1., lf0<=1.], [lambda x:np.exp(x), lambda x:x])    
    return lf0


def pw2wav(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''
    if isinstance(features, dict):
        en = np.reshape(features['en'], [-1, 1])
        sp = np.power(10., features['sp'])
        sp = en * sp
        return pw.synthesize(
            features['f0'].astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            features['ap'].astype(np.float64).copy(order='C'),
            fs,
        )
    features = features.astype(np.float64)
    sp = features[:, :feat_dim]
    ap = features[:, feat_dim:feat_dim*2]
    f0 = features[:, feat_dim*2]
    en = features[:, feat_dim*2 + 1]
    en = np.reshape(en, [-1, 1])
    sp = np.power(10., sp)
    sp = en * sp
    return pw.synthesize(
        f0.copy(order='C'),
        sp.copy(order='C'),
        ap.copy(order='C'),
        fs
    )
    