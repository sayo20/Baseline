import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sklearn
import math, random
import torch
import torchaudio
from torchaudio import transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import os
import cmath
import scipy
from scipy.io.wavfile import write
from scipy import signal
from torch.autograd import Variable
import tqdm
import wandb
from timeit import default_timer as timer
import IPython
import time
from AudioFeatures import InputFeature

class Spatialization():


    def normalize(infile, rms_level=23):
        """
        Normalize the signal given a certain technique (peak or rms).
        Args:
            - infile    (str) : input audio.
            - rms_level (int) : rms level in dB.
            ps: code gotten from Pyaugument, I did not use their package cause they only accept wav format files which my data isnt
        """

        # linear rms level and scaling factor
        # start_time = timer()
        r = 10 ** (rms_level / 10.0)
        a = torch.sqrt((len(infile) * r ** 2) / torch.sum(infile ** 2)) #convert numpy to torch, #time each function to find the bottleneck,get rid of static,store data

        # normalize
        y = infile * a
        # print("--- normalize rms: %s seconds ---" % (timer() - start_time))
        return y

    # def load_audio(audio_path):
    #     #     print(audio_path)
    #     start_time =timer()
    #     aud,sr = torchaudio.load(audio_path)
    #     print("--- load_audio: %s seconds ---" % (timer() - start_time))
    #     return (aud,sr)

    def load_audio(audio_file):
        """
        audio_file: path to the wav file
        """
        # print("audio path",audio_file)
        # start_time =timer()
        signal, sr = torchaudio.load(audio_file)

        #the audios are now 4 seconds long so first we want to try with just 2 seconds (plus i did not want to regenerate 2 second clips so i'll just cut them on the fly)

        two_sec  = signal[:, :sr*2]
        # print("--- load_audio: %s seconds ---" % (timer() - start_time))
        # print("audio shape is: ",signal.shape)
        return (two_sec,sr)
    def convolveBrir(brir_path,sound,threshold,uni_bi):
        """
        brir_path: path to the brir you want to convolve
        sound: the sound you wan to convolve with the brir (convert from tensor to numpy)
        threshold: to prevent the audio from truncating at certain frequenct
        uni_bi: 0 means unilateral so we only one ear of our microphones(so 3ch) and 1: means bilateral, we use all 6
        """
        # start_time = timer()
        brir,sr = InputFeature.load_audio(brir_path)
        # print(sound.is_cuda,brir.is_cuda)
        # sound = sound.to(device)
        # brir = brir.to(device)
        conv_left = signal.convolve(sound[0,:],brir[0,:]) #do like matlab, convolve each ear and then concat
        conv_left = torch.tensor(conv_left) #we have to place them on cuda
        conv_right = signal.convolve(sound[0,:],brir[1,:])
        conv_right = torch.tensor(conv_right)
        conv = 0
        if uni_bi == 0:
            conv = conv_left
            conv = conv.reshape(1,conv.shape[0]).T
        elif uni_bi == 1:
            conv = torch.vstack((conv_left,conv_right)).T
        # print("--- convolve brir: %s seconds ---" % (timer() - start_time))
        return conv *threshold

    def cal_adjusted_rms(clean_rms, snr):
        a = float(snr) / 20
        noise_rms = clean_rms / (10 ** a)
        return noise_rms


    def cal_rms(amp):
        return torch.sqrt(torch.mean(torch.square(amp), axis=-1))

    def mixAudioSnr(clean_amp,noise_amp,snr):
        #https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_mixed_audio_file_with_soundfile.py
        clean_rms = Spatialization.cal_rms(clean_amp)

        start = random.randint(0, len(noise_amp) - len(clean_amp))
        divided_noise_amp = noise_amp[start : start + len(clean_amp)]
        noise_rms = Spatialization.cal_rms(divided_noise_amp)


        adjusted_noise_rms = Spatialization.cal_adjusted_rms(clean_rms, snr)

        adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)

        return clean_amp,adjusted_noise_amp

    def spatializeMixDataset(df,uni_bi):
        """
        pass the df[idx,:]
        uni_bi: to specify uni or bi
        return the mixed convolved audio, target convolved audio, azimuth target and noise, sampling rate
        Then i dont need load_audio cause i will be passing array
        """
        # start_time =timer()
        target_audio,sr = Spatialization.load_audio("../"+df["Target"])
        noise_audio,sr = Spatialization.load_audio("../"+df["Noise"])
        # print()

        snr_t = df["SNR target"]
        snr_n = df["SNR target"] #rms normalize sound
        #rms norm
        target_audio = Spatialization.normalize(target_audio)
        noise_audio= Spatialization.normalize(noise_audio)

        #adjust snr
        norm_target,norm_noise = Spatialization.mixAudioSnr(target_audio,noise_audio,snr_t)
        ####
        az_t = df["Az target"]
        az_n = df["Az noise"]

        brir_num =df["BRIR_no"]
        pairing = df["Mix type"]

        brir_basePath = "../Data/BRIR/"+str(brir_num)+"/"

        brir_target_front = brir_basePath+"CIFrontmicaz"+str(az_t)+"el0.flac"
        brir_target_back = brir_basePath+"CIBackmicaz"+str(az_t)+"el0.flac"
        brir_target_tmic = brir_basePath+"CITmicaz"+str(az_t)+"el0.flac"

        brir_noise_front = brir_basePath+"CIFrontmicaz"+str(az_n)+"el0.flac"
        brir_noise_back = brir_basePath+"CIBackmicaz"+str(az_n)+"el0.flac"
        brir_noise_tmic = brir_basePath+"CITmicaz"+str(az_n)+"el0.flac"

        target_back = Spatialization.convolveBrir(brir_target_back,norm_target,0.2,uni_bi)
        target_front = Spatialization.convolveBrir(brir_target_front,norm_target,0.2,uni_bi)
        target_tmic = Spatialization.convolveBrir(brir_target_tmic,norm_target,0.2,uni_bi)
        # print(f"their shape is {target_back.shape,target_front.shape,target_tmic.shape}")
        convolved_target = torch.cat((target_back,target_front,target_tmic), axis=1) #6-ch audio

        noise_back = Spatialization.convolveBrir(brir_noise_back,norm_noise,0.2,uni_bi)
        noise_front = Spatialization.convolveBrir(brir_noise_front,norm_noise,0.2,uni_bi)
        noise_tmic = Spatialization.convolveBrir(brir_noise_tmic,norm_noise,0.2,uni_bi)
        convolved_noise = torch.cat((noise_back,noise_front,noise_tmic), axis=1) #6-ch audio

        #sum up target and noise
        target_noise_back = target_back + noise_back
        target_noise_front = target_front + noise_front
        target_noise_tmic = target_tmic + noise_tmic

        mixed_audio = torch.cat((target_noise_back,target_noise_front,target_noise_tmic), axis=1)
        # print(f"final shapes are {convolved_target.T.shape,convolved_noise.T.shape,mixed_audio.T.shape}")
        # print("---full spatialization: %s seconds ---" % (timer() - start_time))
        return convolved_target.T, convolved_noise.T, mixed_audio.T, az_t,az_n,sr,snr_t,pairing
