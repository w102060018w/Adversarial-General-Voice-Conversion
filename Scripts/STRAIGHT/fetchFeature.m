clear; clc;

%% Parameters
WINDOWLEN = 25
FRAMESHIFT = 5
FFTLEN = 1024
param = struct();
WAVROOT = 'E:\Dataset\SUPERSEDED - The Voice Conversion Challenge 2016\DS_10283_2042\evaluation_release\'
SAVROOT = ['D:\Lab\CEDL\final\Features\SUPERSEDED - The Voice Conversion Challenge 2016\matlab\evaluation_release-', num2str(WINDOWLEN), '-', num2str(FRAMESHIFT), '-', num2str(FFTLEN), '\']
 if ~(7==exist(SAVROOT))
        mkdir(SAVROOT)
    end

%% Load wav
people = dir(WAVROOT);
for idxPerson = 3:1:length(people)
    person = fullfile(WAVROOT, people(idxPerson).name)
    wavs = dir(person);
    saveDir = fullfile(SAVROOT, people(idxPerson).name);
    if ~(7==exist(saveDir))
        mkdir(saveDir)
    end

    for idxWav = 3:1:length(wavs)
        fileName = strsplit(wavs(idxWav).name, '.');
        fileName = fileName(1);
        wavName = fullfile(WAVROOT, people(idxPerson).name, wavs(idxWav).name);
        savePath = fullfile(saveDir, fileName);
        savePath = savePath{1};

        % Load wav
        [x, fs]=audioread(wavName);
        
        % Extract f0 and ap
        % param=struct('F0defaultWindowLength', WINDOWLEN,  'F0frameUpdateInterval', FRAMESHIFT, 'refineFftLength', FFTLEN);
        [f0raw, ap, analysisParams]=exstraightsource(x, fs,  param);
        
        
        % Extract sp
        % param=struct('defaultFrameLength', WINDOWLEN);
        [n3sgram, nalysisParamsSp]=exstraightspec(x, f0raw, fs, param);
        
        % Save feature
        sizef0 = size(f0raw);
        sizeap = size(ap);
        sizesgram = size(n3sgram);
        minLen = min([sizef0(2), sizeap(2), sizesgram(2)]);
        mat = [f0raw(:,1:minLen); ap(:, 1:minLen); n3sgram(:, 1:minLen)];
        save([savePath, '.mat'], 'mat');
        
    end
end