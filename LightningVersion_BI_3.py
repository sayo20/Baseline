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
# from BaselineModel import DNN
from AudioFeatures import InputFeature
import cProfile, pstats
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
    #     print("Loss in here: ",estimated_mask.shape,target_spectra.shape)
        l= ((estimated_mask - clean_mask)**2 * (mixed_spectra**2))
        loss = l.sum()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=(self.learning_rate),weight_decay=0.0001)#,weight_decay=self.config["Weight_decay"]
        # self.log("learning_rate", self.learning_rate)
        return optimizer

    def training_step(self,batch,batch_idx):
        start_time = timer()
        data, labels,mixed_spectra,mixed_spectra_phase,target_audio,target_spectra,mixed_audio = batch
            #lstm wants shape: [batch,seq length,input shape]
        data = data.reshape(data.shape[0],data.shape[-1],data.shape[1])
        labels = labels.reshape(labels.shape[0],labels.shape[-1],labels.shape[1])
        mixed_spectra = mixed_spectra.reshape(mixed_spectra.shape[0],mixed_spectra.shape[-1],mixed_spectra.shape[1])

        estimated_mask = self(data)
        estimated_label_ = torch.zeros(size=estimated_mask.shape).to(self.device)
        for i in range(estimated_mask.shape[0]):
            estimated_label_[i,:,:] = InputFeature.getPercentNorm(estimated_mask[i,:,:])
        print("norm: ",torch.max(labels),torch.max(estimated_label_))
        loss = self.reconstruction_loss(estimated_mask,labels,mixed_spectra)
        # wandb.log({'reconstruct loss training':loss.item()})
        # print(f'Epoch: {e}\t{100 * (i + 1) / len(self.train_dl):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',end='\r')
        self.log("train_loss", loss,on_epoch=True)
        print("--- training steps takes: %s seconds ---" % (timer() - start_time))
        return loss
    # def training_step_end(self, training_step_outputs):
    #     return {'loss': training_step_outputs['loss'].sum()}
    def validation_step(self,batch,batch_idx):
        data, labels,mixed_spectra,mixed_spectra_phase,target_audio,target_spectra,mixed_audio = batch

        data = data.reshape(data.shape[0],data.shape[-1],data.shape[1])
        labels = labels.reshape(labels.shape[0],labels.shape[-1],labels.shape[1])
        mixed_spectra = mixed_spectra.reshape(mixed_spectra.shape[0],mixed_spectra.shape[-1],mixed_spectra.shape[1])

        estimated_mask = self(data)
        estimated_label_ = torch.zeros(size=estimated_mask.shape).to(self.device)
        for i in range(estimated_mask.shape[0]):
            estimated_label_[i,:,:] = InputFeature.getPercentNorm(estimated_mask[i,:,:])
        # estimated_label_ = estimated_label_.to(se)
        loss = self.reconstruction_loss(estimated_mask,labels,mixed_spectra)
        # wandb.log({'reconstruct loss validation':loss.item()})
        # print(f'Epoch: {e}\t{100 * (i + 1) / len(self.val_dl):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',end='\r')
        self.log("val_loss", loss,on_epoch=True)
        # return loss
    #
    def train_dataloader(self):
        df_train = pd.read_csv("../Data/Libri-speech/Train-20k.csv")
        # df_train = df_train[:32]
        train_ds = SoundDS(df_train,config["Experiment"])
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["workers"],pin_memory=True)
        return train_dl
    def val_dataloader(self):
        df_val = pd.read_csv("../Data/Libri-speech/Val-10k.csv")
        # df_val = df_val[:32]
        val_ds = SoundDS(df_val,config["Experiment"])
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False,num_workers=self.config["workers"],pin_memory=True)
        return val_dl


class ImagePredictionLogger(pl.Callback):
    def __init__(self,val_samples,num_samples=2):
        super().__init__()
        self.val_samples = val_samples
        self.num_samples = num_samples

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
    def pltMasks(self,estimated_mask,true_mask,mixed_spect,target_spect,loss):

        estimated_mask = estimated_mask.reshape(estimated_mask.shape[1],estimated_mask.shape[0])
        true_mask = true_mask.reshape(true_mask.shape[1],true_mask.shape[0])
        mixed_spect = mixed_spect.reshape(mixed_spect.shape[1],mixed_spect.shape[0])
        target_spect = target_spect.reshape(target_spect.shape[1],target_spect.shape[0])


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
        return plt, (estimated_mask,true_mask,mixed_spect,target_spect)

    def on_validation_epoch_end(self,trainer,pl_module):

        self.val_features, self.true_labels,self.val_spectra_mag,self.val_spectra_phase,self.t_audio,self.target_spectra,self.mixed_audio = self.val_samples
        self.true_labels = self.true_labels[:self.num_samples,:,:]
        self.val_features = self.val_features[:self.num_samples,:,:]
        self.val_spectra_mag = self.val_spectra_mag[:self.num_samples,:,:]
        self.target_spectra = self.target_spectra[:self.num_samples,:,:]
        self.mixed_audio = self.mixed_audio[:self.num_samples,:,:]
        self.t_audio = self.t_audio[:self.num_samples,:,:]
        self.val_spectra_phase = self.val_spectra_phase[:self.num_samples,:,:]

        true_labels = self.true_labels.to(device="cpu")
        mixed_audios = self.val_features.to(device=pl_module.device)
        self.val_spectra_mag = self.val_spectra_mag.to(device ="cpu" )
        self.target_spectra = self.target_spectra.to(device ="cpu")

        true_labels = true_labels.reshape(true_labels.shape[0],true_labels.shape[-1],true_labels.shape[1])
        mixed_audios = mixed_audios.reshape(mixed_audios.shape[0],mixed_audios.shape[-1],mixed_audios.shape[1])
        self.val_spectra_mag = self.val_spectra_mag.reshape(self.val_spectra_mag .shape[0],self.val_spectra_mag.shape[-1],self.val_spectra_mag .shape[1])
        self.target_spectra = self.target_spectra.reshape(self.target_spectra .shape[0],self.target_spectra.shape[-1],self.target_spectra .shape[1])

        estimated_label = pl_module(mixed_audios).cpu()

        loss,avg_loss = self.reconstruction_loss(estimated_label,true_labels,self.val_spectra_mag)
        arrays_ = {"Mixed audio":[],"Target audio":[],"Target spectra":[],"Mixed spectra":[],"Mixed phase":[],"Estimated mask":[],"Target mask":[],"Loss":[],"Snr target":[],"Snr noise":[],"Azimuth target":[],"Azimuth noise":[]}
        i=0
        examples=[]
        print("true labels: ",true_labels.shape)
        df = pd.DataFrame(columns=["Mixed audio","Target audio","Target spectra","Mixed spectra","Mixed phase","Estimated mask","True mask","Loss"])

        for estimated,true,mixed_spect,mixed_phase,target_spect,t_audio,mix_audio,losses in zip(estimated_label,true_labels,self.val_spectra_mag,self.val_spectra_phase,self.target_spectra,self.t_audio ,self.mixed_audio,avg_loss):

            img,(estimated_mask,true_mask,mixed_spect_,target_spect_) = self.pltMasks(estimated,true,mixed_spect,target_spect,losses)
            examples.append(wandb.Image(img))
            img.close()

            df.loc[i,"Mixed audio"] = (mix_audio) #estimated_label,true_labels,self.val_spectra_mag,self.val_spectra_phase,self.target_spectra,self.t_audio ,self.mixed_audio,avg_loss
            df.loc[i,"Target audio"] = (t_audio)
            df.loc[i,"Mixed spectra"] = (target_spect_)
            df.loc[i,"Target spectra"] = (mixed_spect)
            df.loc[i,"Mixed phase"] = (mixed_phase)
            df.loc[i,"Estimated mask"] = (estimated_mask)
            df.loc[i,"True mask"] = (true_mask)
            df.loc[i,"Loss"] = losses

            i= i+1
        trainer.logger.experiment.log({
            "examples":examples
        })
        # df.to_pickle("PlotSanityCheck_lpsMask-lps-3.pkl")

        df.to_pickle("PlotSanityCheck_lpsMask-bi-3.pkl")


if __name__ == '__main__':
    ####INFO: RUN 1: Normalized mixed spectra nd target spectra and  unbounded max
    wandb_logger = WandbLogger(project="Exploring-Baseline-Experiment3", config='config_BI_3.yaml',name="bi-GU-LPS-V2-Ex3")#Bilateral_gu_LpsMasks
    # run = wandb.init(project="NeuralFilter", config='config.yaml')

    config = wandb_logger.experiment.config
    # os.environ["CUDA_VISIBLE_DEVICES"] ="1" #config["CUDA_VISIBLE_DEVICES"]
    #Hyper params
    df_train = pd.read_csv("../Data/Libri-speech/Train-20k.csv")#20 for m-m,m-f,f-f so total 60k
    df_val = pd.read_csv("../Data/Libri-speech/Val-10k.csv")

    # train_ds = SoundDS(df_train,config["Experiment"],"cuda") #0 for unilateral, 1 for bilateral
    print("before val_dl",config["Experiment"])
    val_ds = SoundDS(df_val,config["Experiment"])

    # Create training and validation data loaders
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"],pin_memory=True)#64
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"],num_workers=config["workers"],shuffle=False)#64

    val_samples = next(iter(val_dl))
    print("after val_dl,",val_samples[0][0,:,:].shape)
    input_size = config["Input_bi"] #train_features.shape[1]#1799 1542
    # print(f"input size is {input_size.shape}")
    no_lstm_node = config["Number_of_nodes"]#512
    no_output_node = config["Label_bi"] #train_labels.shape[1]#257 not cuz we have 6 channels
    no_lstm_layers=config["Number_of_layers"]#3
    epoch = config["epochs"]
    # print(f"input size is {input_size,train_labels.shape[1]}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config["Patience"], verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="Bi_experiments/",
        save_top_k =-1,
        filename="bi-GU-LPS-V2-Ex3-{epoch:02d}-{val_loss:.2f}",#-{epoch:02d}-{val_loss:.2f}
        mode="min",)

    # trainer = Trainer(max_epochs=epoch,fast_dev_run=False,gpus=1,callbacks=[early_stop_callback,checkpoint_callback],logger=wandb_logger,auto_lr_find=True,deterministic=True, profiler="simple")
    trainer = Trainer(max_epochs=epoch,fast_dev_run=False,gpus=[1],callbacks=[early_stop_callback,checkpoint_callback,ImagePredictionLogger(val_samples)],logger=wandb_logger,auto_lr_find=True, profiler="simple") # max_epochs=epoch ,profiler="advanced", runs one batch to test that the model works
    model = DNN(input_size, no_lstm_node, no_output_node, no_lstm_layers,config)
    print("started fitting, model on gpu: ")
    trainer.fit(model = model)#trainer.fit(model = model,ckpt_path="Data/bi_model_bestrun.ckpt")
    # trainer.validate(dataloaders=val_dl,model = model,ckpt_path="Bi_experiments/Bilateral_gu_v1.ckpt")
    print("stopped fitting")
    trainer.save_checkpoint("Bi_experiments/bi-GU-LPS-V2-Ex3.ckpt")
    script = model.to_torchscript()
    torch.jit.save(script, "Bi_experiments/bi-GU-LPS-V2-Ex3.pt")





