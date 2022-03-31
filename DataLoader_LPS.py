
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
from torch.autograd import Variable
import tqdm
import wandb
from timeit import default_timer as timer
import IPython
from AudioFeatures import InputFeature
from Spatialization import Spatialization
from timeit import  Timer
import time


class SoundDS_LPS(Dataset):

    def __init__(self, df,type):
        self.df = df
        self.type = type
        # self.device = device

    #-------------------------------
    #Number of items in dataset
    #-------------------------------
    def __len__(self):
        return len(self.df)

  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
    def __getitem__(self, idx):
        # print("im here")
        #load audio
        # start_time = timer()
        target_audio, noise_audio, mixed_audio, az_t,az_n,sr,snr,pairing = Spatialization.spatializeMixDataset(self.df.loc[idx,:],self.type)
        # print("--- spatialization took: %s seconds ---" % (timer() - start_time))
        # print("sr is: ",sr)
        # get spectral of the first channel
        # target_audio = torch.tensor(target_audio)
        # noise_audio = torch.tensor(noise_audio)
        # mixed_audio = torch.tensor(mixed_audio)
        # lps_time = timer()
        spectral,sr = InputFeature.getLPS_firstchannel(mixed_audio,0)
        # spectral = InputFeature.getPercentNorm(spectral)
        # print("--- lps of first channel: : %s seconds ---" % (timer() - lps_time))
        #get spatial feature

        #input feature
        # ipd_time = timer()
        # if self.type == 0:
        #     spatial = InputFeature.getSpatialFeature_uni(mixed_audio)
        #     input_ = np.concatenate((spectral,spatial)) ##the firs 3 ipd is on the left ear ch 1,3,5
        # elif self.type == 1:
        #     uni_ipd,bi_ipd = InputFeature.getSpatialFeature(mixed_audio)
        #     input_ = np.concatenate((spectral,bi_ipd))
        # print("--- ipd: : %s seconds ---" % (timer() - ipd_time))
#         print(f"{spectral.shape,uni_ipd[:772].shape,input_.shape}")
        #label: this should be the tf-mask of the target audio since thats what we want to estimate

        # traget_spectra_mag =InputFeature.getLogSpectra(target_audio) #InputFeature.getLPS_firstchannel(target_audio,0) #InputFeature.getPercentNorm(InputFeature.getLPS_firstchannel(target_audio,0)[0])
        # mixed_spectra_mag=InputFeature.getLogSpectra(mixed_audio)

        traget_spectra_mag,sr = InputFeature.getLPS_firstchannel(target_audio,0)
        mixed_spectra_mag,sr=InputFeature.getLPS_firstchannel(mixed_audio,0)
        # traget_spectra_mag = InputFeature.getPercentNorm(traget_spectra_mag)
        # mixed_spectra_mag = InputFeature.getPercentNorm(mixed_spectra_mag)


        # traget_spectra_mag,sr = InputFeature.getLPS_firstchannel(target_audio,0) #un normalized
        # traget_spectra_mag =torch.abs(InputFeature.getLogPowerSpectrum(target_audio[0,:]))

        # traget_spectra_mag = torch.abs(traget_spectra_mag_)
        # traget_spectra_mag = InputFeature.getPercentNorm(traget_spectra_mag)
        # traget_spectra_mag = traget_spectra_mag.reshape(traget_spectra_mag.shape[0]*traget_spectra_mag.shape[1],traget_spectra_mag.shape[2])

        # mixed_spectra_mag = torch.abs(InputFeature.getLogPowerSpectrum(mixed_audio[0,:]))
         #InputFeature.getLPS_firstchannel(mixed_audio,0)#InputFeature.getPercentNorm(InputFeature.getLPS_firstchannel(mixed_audio,0)[0]) #no need to take abs again since lps gives us the log magnitude already
        # mixed_spectra_mag,sr = InputFeature.getLPS_firstchannel(mixed_audio,0)

        # mixed_spectra_mag = torch.abs(mixed_spectra_)
        # mixed_spectra_mag = InputFeature.getPercentNorm(mixed_spectra_mag)
        # mixed_spectra_mag = mixed_spectra_mag.reshape(mixed_spectra_mag.shape[0]*mixed_spectra_mag.shape[1],mixed_spectra_mag.shape[2])

        mixed_spectra_phase = InputFeature.getLogPowerSpectrum(mixed_audio)
        mixed_spectra_phase = torch.angle(mixed_spectra_phase)[0,:,:]
        # mixed_spectra_phase = InputFeature.getPercentNorm(mixed_spectra_phase)
        # mixed_spectra_phase = mixed_spectra_phase.reshape(mixed_spectra_phase.shape[0]*mixed_spectra_phase.shape[1],mixed_spectra_phase.shape[2])


        ssm_mask  = traget_spectra_mag/mixed_spectra_mag
        # print("mask shapes: ",torch.min(ssm_mask),torch.max(ssm_mask))
        # ssm_mask[ssm_mask > ]=
        # ssm_mask = InputFeature.getPercentNorm(ssm_mask)
        # print("--- data loading took: : %s seconds ---" % (timer() - start_time))
        # print(f"label shape is {ssm_mask.shape}")
        return spectral,ssm_mask,mixed_spectra_mag,mixed_spectra_phase,target_audio,traget_spectra_mag,mixed_audio,az_t,az_n,snr,pairing
            # spectral,ssm_mask,mixed_spectra_mag,mixed_spectra_phase,target_audio,traget_spectra_mag,mixed_audio,az_t,az_n,snr,pairing
#         return 0,0,0,0,0

