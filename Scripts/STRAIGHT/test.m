clear; clc;

%% Parameters
WINDOWLEN = 25;
FRAMESHIFT = 5;
FFTLEN = 1024;
param = struct();

%% Load wav
fname = './wav/SM1/100002.wav';
[x, fs]=audioread(fname);

%% Extract f0 and ap
% param=struct('F0defaultWindowLength', WINDOWLEN,  'F0frameUpdateInterval', FRAMESHIFT, 'refineFftLength', FFTLEN);
% param=struct('F0defaultWindowLength', WINDOWLEN,  'F0frameUpdateInterval', FRAMESHIFT);
[f0raw, ap, analysisParams]=exstraightsource(x, fs,  param);

%% Extract sp
% param=struct('defaultFrameLength', WINDOWLEN);
[n3sgram, nalysisParamsSp]=exstraightspec(x, f0raw, fs, param);

%% Speech Synthesis
% param=struct('spectralUpdateInterval', FRAMESHIFT);
[sy, prmS] = exstraightsynth(f0raw, n3sgram, ap, fs);
mean = mean(sy);
std = std(sy);
sy_norm =  (sy-mean)/std;

%% Play
% sound(x,fs)
% sound(sy_norm, fs)
