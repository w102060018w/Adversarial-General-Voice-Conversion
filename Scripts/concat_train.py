#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:52:53 2017

@author: chadyang
"""

import os
from glob import glob
import joblib
from tqdm import tqdm
import numpy as np
from collections import defaultdict

#%% parameters
TRAINFEAROOT = '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/vcc2016_training/'
TESTFEAROOT = '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/evaluation_release/'
NUMCLASS = 10

#%% concat
def one_hot_encode(labels, class_n):
    n_labels = len(labels)
    # n_unique_labels = len(np.unique(labels))
    n_unique_labels = class_n
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

trainFea = []
trainLabel = []
trainFea2 = []
trainLabel2 = []
trainFea2All = []
for speaker in sorted(os.listdir(TRAINFEAROOT)):
    print(speaker)
    for frame in sorted(glob(os.path.join(TRAINFEAROOT, speaker, '*.pkl'))):
        data = joblib.load(frame)
        fea = data[:,:513] # spectral feature dim=0:513
        label = data[:,-1]
        trainFea.append(fea)
        trainLabel.append(label)
        if speaker in ['SF1', 'TM3']:
            trainFea2.append(fea)
            trainLabel2.append(label)
            trainFea2All.append(data)
trainFea2All = np.concatenate(trainFea2All)
joblib.dump(trainFea2All, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainFea-2people-all.pkl')

            
trainFea = np.concatenate(trainFea, axis=0)
trainLabel = np.concatenate(trainLabel, axis=0).astype(int)
trainLabel1hot = one_hot_encode(trainLabel.tolist(), NUMCLASS)
joblib.dump(trainFea, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainFea.pkl')
joblib.dump(trainLabel1hot, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainLabel.pkl')


trainFea2 = np.concatenate(trainFea2, axis=0)
trainLabel2 = np.concatenate(trainLabel2, axis=0).astype(int)
trainLabel2[trainLabel2==9] = 1
trainLabel1hot2 = one_hot_encode(trainLabel2.tolist(), 2)
joblib.dump(trainFea2, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainFea-2people.pkl')
joblib.dump(trainLabel1hot2, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/trainLabel-2people-2.pkl')


#%%
testFea = []
testLabel = []
for speaker in tqdm(sorted(os.listdir(TESTFEAROOT))):
    print(speaker)
    for frame in sorted(glob(os.path.join(TESTFEAROOT, speaker, '*.pkl'))):
        data = joblib.load(frame)
        fea = data[:,:513] # spectral feature dim=0:513
        label = data[:,-1]
        testFea.append(fea)
        testLabel.append(label)
testFea = np.concatenate(testFea, axis=0)
testLabel = np.concatenate(testLabel, axis=0).astype(int)
testLabel1hot = one_hot_encode(testLabel.tolist(), NUMCLASS)

joblib.dump(testFea, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/testFea.pkl')
joblib.dump(testLabel1hot, '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/testLabel.pkl')


#%% fo stats
SPEAKERPATH = '/home/chadyang/CEDL/final/Scripts/SPEAKERS.tsv'
with open(SPEAKERPATH, 'r') as f:
    SPEAKERS = f.readlines()
SPEAKERS = [s.split('\n')[0] for s in SPEAKERS]

f0dict = {}
for speaker in SPEAKERS:
    print(speaker)
    f0s = []
    for frame in sorted(glob(os.path.join(TRAINFEAROOT, speaker, '*.pkl'))):
        data = joblib.load(frame)
        f0 = data[:,1027]
        f0s.extend(f0)
    if len(f0s)==0:
        break
    f0s = np.asarray(f0s)
    f0s = f0s[f0s > 2.]
    f0s = np.log(f0s)
    mu, std = f0s.mean(), f0s.std()
    # Save as `float32`
    savpath = '/home/chadyang/CEDL/final/Features/SUPERSEDED-The-Voice-Conversion-Challenge-2016/pkl2/f0/{}.pkl'.format(speaker)
    joblib.dump(np.asarray((mu, std)), savpath)
    