import pytorch_lightning as pl
from pytorch_lightning import Trainer
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
import json
from DataLoader import SoundDS
from DataLoader_inference import SoundDS_T
# from BaselineModel import DNN
from AudioFeatures import InputFeature
import cProfile, pstats
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio,signal_distortion_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

# wandb.login()
class DNN(pl.LightningModule):

    def __init__(self, input_size, no_lstm_node, no_output_node, no_lstm_layers,config):
        super(DNN,self).__init__()

        self.input_size= input_size
        self.no_lstm_node = no_lstm_node
        self.no_output_node = no_output_node
        self.no_lstm_layers = no_lstm_layers
        self.config = config
        self.learning_rate = self.config["lr"]

        #3lstm layers
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = no_lstm_node, num_layers = no_lstm_layers ,batch_first = True)
        self.fc = nn.Linear(no_lstm_node,no_output_node)


    def forward(self,x):

        # h_0 = Variable(torch.zeros(self.no_lstm_layers, x.size(0), self.no_lstm_node)).to(self.device) #output frm previous block
        # c_0 = Variable(torch.zeros(self.no_lstm_layers, x.size(0), self.no_lstm_node)).to(self.device) #memory from previous block

        output, (h,c) = self.lstm(x)
#         print(output[:,-1,:].shape)
        out = self.fc(output)
        # print("models grad: ",out.grad_fn)
        # print(self.parameters())
        return out


    def reconstruction_loss(self,estimated_mask,clean_mask,mixed_spectra):

        """
        clean_mask: the cleaned spectrogram of the target speaker
        estimated_mask: the spectrogram of the mask estimated by the network
        mixed_spectra: the magnitude spectrogram of the mixed input to the model

        return: signal reconstruction loss
        """
        all_loss = []
        # arrays_ = {"Mixed spectra":[],"Target Spectra":[],"Estimated mask":[],"Target mask":[],"Mixed audio":[],"Target audio":[],"Mixed phase":[]}
        arrays_ = {"Mixed audio":[],"Target audio":[],"Target spectra":[],"Mixed spectra":[],"Mixed phase":[],"Estimated mask":[],"Target mask":[],"Loss":[],"Snr target":[],"Snr noise":[],"Azimuth target":[],"Azimuth noise":[]}

        for i in range(estimated_mask.shape[0]):

            l = ((estimated_mask[i,:,:] - clean_mask[i,:,:])**2) * (mixed_spectra[i,:,:]**2)
            sum_per_sample = l.sum()
            all_loss.append(sum_per_sample)

        loss = ((estimated_mask - clean_mask)**2 * (mixed_spectra**2)).sum()
        return loss,all_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=(self.learning_rate),weight_decay=0.01)
        # self.log("learning_rate", self.learning_rate)
        return optimizer


    def revertToAudio(self,mixed_spect,estimated_mask,mixed_phase,mixed_audio):
        # estimated_mask = estimated_mask.reshape(estimated_mask.shape[1],estimated_mask.shape[0])
        # mixed_spect =  estimated_mask.reshape(mixed_spect.shape[1],mixed_spect.shape[0])
        est_maks = InputFeature.Db2linear(estimated_mask)
        mixed_spectra = InputFeature. Db2linear(mixed_spect)
        print(est_maks.shape,mixed_spectra.shape,mixed_phase.shape)
        #get and save target and est target
        masked_spec=  est_maks * mixed_spectra
        masked_stft = masked_spec * torch.exp(1j * mixed_phase)
        targ_aud  = torch.istft(masked_stft,512,length=mixed_audio.shape[-1])

        return targ_aud

    def pltMasks(self,estimated_mask,true_mask,mixed_spect,target_spect,loss):
        # print("plot shapes",estimated_mask.shape,true_mask.shape,mixed_spect.shape,target_spect.shape)
        # print(f"estimated mask min max in plt: {torch.min(estimated_mask),torch.max(estimated_mask)}")
        # print(f"true mask min max in plt: {torch.min(true_mask),torch.max(true_mask)}")
        # print(f"mixed mask min max in plt: {torch.min(mixed_spect),torch.max(mixed_spect)}")
        # estimated_mask = estimated_mask.reshape(estimated_mask.shape[1],estimated_mask.shape[0])
        # true_mask = true_mask.reshape(true_mask.shape[1],true_mask.shape[0])
        # mixed_spect = mixed_spect.reshape(mixed_spect.shape[1],mixed_spect.shape[0])
        # target_spect = target_spect.reshape(target_spect.shape[1],target_spect.shape[0])


        # print("true mask * mixed mask in plt: ",torch.min(true_mask * mixed_spect),torch.max(true_mask * mixed_spect))
        fig, ax = plt.subplots(1, 6, figsize=(15, 4))

        mixed_spect_ = ax[0].imshow(InputFeature.getPercentNorm(mixed_spect),origin='lower', aspect='auto',cmap='jet')#20 * torch.log10 vmin=0, vmax=5,cmap='jet,aspect='auto'
        ax[0].set_title("Mixed Spect(MS)")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Frequency")
        plt.colorbar(mixed_spect_, ax=ax[0])

        target_spect_ = ax[1].imshow(InputFeature.getPercentNorm(target_spect),origin='lower', aspect='auto',cmap='jet')
        ax[1].set_title("Target Spect(TS)")
        ax[1].label_outer()
        plt.colorbar(target_spect_, ax=ax[1])

        true_mask_ = ax[2].imshow(InputFeature.getPercentNorm(true_mask),origin='lower', aspect='auto',cmap='jet')#20 * torch.log10
        ax[2].set_title("True mask(TM)")
        ax[2].label_outer()
        plt.colorbar(true_mask_, ax=ax[2])

        estimated_mask_ = ax[3].imshow(InputFeature.getPercentNorm(estimated_mask),origin='lower',aspect='auto',cmap='jet')
        ax[3].set_title("Est mask(EM),"+str(math.floor(loss.item())))#
        ax[3].label_outer()
        plt.colorbar(estimated_mask_, ax=ax[3])

        estimated_mask_mixed = ax[5].imshow(InputFeature.getPercentNorm(estimated_mask * mixed_spect),origin='lower', aspect='auto',cmap='jet')
        ax[5].set_title('EM*MS')
        ax[5].label_outer()
        plt.colorbar(estimated_mask_mixed, ax=ax[5])

        true_mask_mixed = ax[4].imshow(InputFeature.getPercentNorm(true_mask * mixed_spect),origin='lower', aspect='auto',cmap='jet')
        ax[4].set_title('TM *MS')
        ax[4].label_outer()
        plt.colorbar(true_mask_mixed, ax=ax[4])
        plt.show()
        # print("done ploting")
        return plt
    def test_step(self,batch,batch_idx):
        print("batch index is: ",batch_idx)
        data, labels,mixed_spectra,mixed_spectra_phase,target_audio,target_spectra,mixed_audio,az_t,az_n,snr,pairing = batch
        pairing = list(pairing)
        df_test = pd.DataFrame(columns=["Mixed audio","Target audio", "Estimated audio","True mask","Estimated mask","Mixed spectra","Target spectra","Mixed phase","Az t","Az n", "snr","Mix type","reconstruction loss","si-sdr","si-snr","stoi","sdr","pesq"])

        data = data.reshape(data.shape[0],data.shape[-1],data.shape[1])
        labels = labels.reshape(labels.shape[0],labels.shape[-1],labels.shape[1])
        mixed_spectra = mixed_spectra.reshape(mixed_spectra.shape[0],mixed_spectra.shape[-1],mixed_spectra.shape[1])
        target_spectra  = target_spectra.reshape(target_spectra.shape[0],target_spectra.shape[-1],target_spectra.shape[1])
        # mixed_spectra_phase =

        estimated_mask = self(data)
        loss,all_loss = self.reconstruction_loss(estimated_mask,labels,mixed_spectra)
        labels = labels.reshape(labels.shape[0],labels.shape[-1],labels.shape[1]).cpu()


        examples=[]
        mixed_spectra = mixed_spectra.reshape(mixed_spectra.shape[0],mixed_spectra.shape[-1],mixed_spectra.shape[1])
        estimated_mask = estimated_mask.reshape(estimated_mask.shape[0],estimated_mask.shape[-1],estimated_mask.shape[1])
        target_spectra  = target_spectra.reshape(target_spectra.shape[0],target_spectra.shape[-1],target_spectra.shape[1])

        # print(f"len of data {len(data)}")
        # print("len of all: ",estimated_mask.shape,mixed_spectra.shape,target_spectra.shape)

        for i in range(len(data)):#cause is batch size len(data)
            predicted_audio = InputFeature.revertToAudio(mixed_spectra[i,:,:],estimated_mask[i,:,:],mixed_spectra_phase[i,:,:],mixed_audio)
            true_audio = target_audio[i,:]
            print(predicted_audio.shape,true_audio.shape,target_audio.shape,estimated_mask[i,:,:].shape,len(batch))
            sisdr =  scale_invariant_signal_distortion_ratio(predicted_audio,true_audio[0,:])#we get a single channeled audio since our mask is single channeled
            sisnr = scale_invariant_signal_noise_ratio(predicted_audio,true_audio[0,:])
            # snr = signal_noise_ratio(predicted_audio,true_audio[0,:])
            stoi = short_time_objective_intelligibility(predicted_audio,true_audio[0,:],16000)
            sdr = signal_distortion_ratio(predicted_audio,true_audio[0,:])
            pesq = perceptual_evaluation_speech_quality(predicted_audio,true_audio[0,:],16000,'nb')

            print("sisdr: ",sisdr,stoi,sdr,pesq)

            mixed_spectra = mixed_spectra.cpu()
            estimated_mask = estimated_mask.cpu()
            target_spectra  = target_spectra.cpu()
            mixed_spectra_phase = mixed_spectra_phase.cpu()
            mixed_audio = mixed_audio.cpu()
            target_audio = target_audio.cpu()
            mixed_spectra_phase =mixed_spectra_phase.cpu()
            all_loss_ = all_loss[i].cpu()
            az_t = az_t.cpu()
            az_n = az_n.cpu()
            snr = snr.cpu()
            # pairing = pairing.cpu()
            loss = loss.cpu()
            sisdr = sisdr.cpu()
            sisnr = sisnr.cpu()
            stoi = stoi.cpu()
            sdr = sdr.cpu()
            pesq = pesq.cpu()
            predicted_audio = predicted_audio.cpu()

            df_test.loc[i, "Mixed audio"] =(mixed_audio[i,:])
            df_test.loc[i, "Target audio"] = target_audio[i,:]
            df_test.loc[i, "Estimated audio"] = predicted_audio

            df_test.loc[i, "True mask"] = labels[i,:,:]
            # print("mask: ",estimated_mask.shape)
            df_test.loc[i, "Estimated mask"] = estimated_mask[i,:,:]
            df_test.loc[i, "Target spectra"] = target_spectra[i,:,:]
            df_test.loc[i, "Mixed spectra"] = mixed_spectra[i,:,:]
            df_test.loc[i, "Mixed phase"] = mixed_spectra_phase[i,:,:]
            df_test.loc[i, "Az t"] = az_t[i]
            df_test.loc[i, "Az n"] = az_n[i]
            df_test.loc[i, "snr"] = snr[i]
            df_test.loc[i, "Mix type"] = pairing[i]
            df_test.loc[i, "reconstruction loss"] = all_loss_
            df_test.loc[i, "si-sdr"] = sisdr
            df_test.loc[i, "si-snr"] = sisnr
            df_test.loc[i, "stoi"] = stoi
            df_test.loc[i, "sdr"] = sdr
            df_test.loc[i, "pesq"] = pesq

        #     img= self.pltMasks(estimated_mask[i,:,:],labels[i,:,:],mixed_spectra[i,:,:],target_spectra[i,:,:],all_loss_)
        #     examples.append(wandb.Image(img))
        #     img.close()
        # trainer.logger.experiment.log({
        #     "examples":examples
        # })
        # df_test.to_pickle("Uni_experiments/uni-GU-LPS-V2-Ex1-TEST-"+str(batch_idx)+".pkl")
        df_test_dict =df_test.to_dict()
        torch.save(df_test_dict,"Uni_experiments_2-results/uni-GU-LPS-V2-Ex2-TEST-"+str(batch_idx)+".pt")

    def test_dataloader(self):
        df_test = pd.read_csv("../Data/Libri-speech/Test-1k.csv")
        # df_val = df_val[:32]
        test_ds = SoundDS_T(df_test,config["Experiment"])
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False,num_workers=self.config["workers"])
        return test_dl
if __name__ == '__main__':
    wandb_logger = WandbLogger(project="Exploring-Baseline-Experiment2", config='config_UNI.yaml',name="uni-GU-LPS-V2-Ex2-TEST")#withouth window length
    # run = wandb.init(project="NeuralFilter", config='config.yaml')

    config = wandb_logger.experiment.config
    # os.environ["CUDA_VISIBLE_DEVICES"] ="1" #config["CUDA_VISIBLE_DEVICES"]
    #Hyper params
    df_train = pd.read_csv("../Data/Libri-speech/Train-20k.csv")#20 for m-m,m-f,f-f so total 60k
    df_val = pd.read_csv("../Data/Libri-speech/Val-10k.csv")

    input_size = config["Input_uni"] #train_features.shape[1]#1799 1542

    no_lstm_node = config["Number_of_nodes"]#512
    no_output_node = config["Label_uni"] #train_labels.shape[1]#257 not cuz we have 6 channels
    no_lstm_layers=config["Number_of_layers"]#3
    epoch = config["epochs"]
    # print(f"input size is {input_size,train_labels.shape[1]}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config["Patience"], verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="Uni_experiments/",
        filename="uni-GU-LPS-V2-Ex2-{epoch:02d}-{val_loss:.2f}",#
        mode="min",)

    # trainer = Trainer(max_epochs=epoch,fast_dev_run=False,gpus=1,callbacks=[early_stop_callback,checkpoint_callback],logger=wandb_logger,auto_lr_find=True,deterministic=True, profiler="simple")
    trainer = Trainer(max_epochs=epoch,fast_dev_run=False,gpus=[3],logger=wandb_logger,auto_lr_find=True, profiler="simple") # max_epochs=epoch ,profiler="advanced", runs one batch to test that the model works
    model = DNN(input_size, no_lstm_node, no_output_node, no_lstm_layers,config)
    print("started fitting, model on gpu: ")
    trainer.test(model=model,ckpt_path = config["Best_epoch"])



