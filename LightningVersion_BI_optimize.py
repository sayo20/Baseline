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
from DataLoader_LPS import SoundDS_LPS
# from BaselineModel import DNN
from AudioFeatures import InputFeature
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch import optim
import cProfile, pstats
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# wandb.login()
class DNN(pl.LightningModule):

    def __init__(self, input_size, no_lstm_node, no_output_node, no_lstm_layers,batch_size,optimizer,lr,wd,config):
        super(DNN,self).__init__()

        self.input_size= input_size
        self.no_lstm_node = no_lstm_node
        self.no_output_node = no_output_node
        self.no_lstm_layers = no_lstm_layers
        self.config = config
        self.learning_rate = lr
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.wd = wd

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
        optimizer = self.optimizer(self.parameters(),lr=(self.learning_rate),weight_decay = )
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
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.config["workers"],pin_memory=True)
        return train_dl
    def val_dataloader(self):
        df_val = pd.read_csv("../Data/Libri-speech/Val-10k.csv")
        # df_val = df_val[:32]
        val_ds = SoundDS(df_val,config["Experiment"])
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,num_workers=self.config["workers"],pin_memory=True)
        return val_dl



#: optuna.trial.Trial
def objective(trial):
    #config['lr'][0],config['lr'][1]
    params = {
          'learning_rate': trial.suggest_loguniform('learning_rate',1e-5,1e-1 ),
          'optimizer': trial.suggest_categorical("optimizer", config['optimizers']),
          'n_unit': trial.suggest_int("n_unit", config['Number_of_nodes'][0], config['Number_of_nodes'][1],config['Number_of_nodes'][2]),
          'n_layer': trial.suggest_int("n_layer", config['Number_of_layers'][0], config['Number_of_layers'][1]),
          'batch_size': trial.suggest_int("batch_size",config['batch_size_range'][0],config['batch_size_range'][1],config['batch_size_range'][2]),
          'weight_decay':trial.suggest_int("weight_decay",config['weight_decay'][0],config['weight_decay'][1],config['weight_decay'][2]),
          }
    optimizer = getattr(optim,params["optimizer"])
    model = DNN(config["Input_bi"], params["n_unit"], config["Label_bi"],params["n_layer"] ,params["batch_size"],optimizer,params["learning_rate"],params["weight_decay"],config)

    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="Bi_experiments-optim/",
    save_top_k =-1,
    filename="bi-GU-LPS-V2-ExOptim-{epoch:02d}-{val_loss:.2f}",#-{epoch:02d}-{val_loss:.2f}
    mode="min")

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config["Patience"], verbose=False, mode="min")
    trainer = Trainer(max_epochs=config["epochs"],fast_dev_run=False,gpus=[1],callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"),early_stop_callback],logger=wandb_logger) # max_epochs=epoch ,profiler="advanced", runs one batch to test that the model works
    trainer.logger.log_hyperparams(params)
    trainer.fit(model = model)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == '__main__':
    ####INFO: RUN 1: Normalized mixed spectra nd target spectra and  unbounded max
    wandb_logger = WandbLogger(project="Exploring-Baseline-Experiment-optim", config='config_BI-optimized.yaml',name="bi-GU-LPS-V2-Ex2-optim")#Bilateral_gu_LpsMasks
    # run = wandb.init(project="NeuralFilter", config='config.yaml')

    config = wandb_logger.experiment.config


    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=config["n_trials"])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #save best tial and log visualizations
    torch.save(trial,config["save_path"])
    trainer.logger.experiment.log({
        "Val_loss":wandb.Image(optuna.visualization.plot_intermediate_values(study)),
        "history":wandb.Image(optuna.visualization.plot_optimization_history(study)),
        "param combo": wandb.Image(optuna.visualization.plot_parallel_coordinate(study)),
        "param importance": wandb.Image(optuna.visualization.plot_param_importances(study))
        })

    wandb_logger.log(trial)







    # os.environ["CUDA_VISIBLE_DEVICES"] ="1" #config["CUDA_VISIBLE_DEVICES"]
    #Hyper params
    # df_train = pd.read_csv("../Data/Libri-speech/Train-20k.csv")#20 for m-m,m-f,f-f so total 60k
    # df_val = pd.read_csv("../Data/Libri-speech/Val-10k.csv")
    #
    # # train_ds = SoundDS(df_train,config["Experiment"],"cuda") #0 for unilateral, 1 for bilateral
    # print("before val_dl",config["Experiment"])
    # val_ds = SoundDS(df_val,config["Experiment"])
    #
    # # Create training and validation data loaders
    # # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"],pin_memory=True)#64
    # val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"],num_workers=config["workers"],shuffle=False)#64
    #
    # val_samples = next(iter(val_dl))
    # print("after val_dl,",val_samples[0][0,:,:].shape)
    # input_size = config["Input_bi"] #train_features.shape[1]#1799 1542
    # # print(f"input size is {input_size.shape}")
    # no_lstm_node = config["Number_of_nodes"]#512
    # output_size = config["Label_bi"] #train_labels.shape[1]#257 not cuz we have 6 channels
    # no_lstm_layers=config["Number_of_layers"]#3
    # epoch = config["epochs"]
    # # print(f"input size is {input_size,train_labels.shape[1]}")
    # early_stop_callback = EarlyStopping(monitor="val_loss", patience=config["Patience"], verbose=False, mode="min")
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",
    #     dirpath="Bi_experiments/",
    #     save_top_k =-1,
    #     filename="bi-GU-LPS-V2-Ex2-logmask-{epoch:02d}-{val_loss:.2f}",#-{epoch:02d}-{val_loss:.2f}
    #     mode="min",)
    #
    # # trainer = Trainer(max_epochs=epoch,fast_dev_run=False,gpus=1,callbacks=[early_stop_callback,checkpoint_callback],logger=wandb_logger,auto_lr_find=True,deterministic=True, profiler="simple")
    # trainer = Trainer(max_epochs=epoch,fast_dev_run=False,gpus=[1],callbacks=[early_stop_callback,checkpoint_callback,ImagePredictionLogger(val_samples)],logger=wandb_logger,auto_lr_find=True, profiler="simple") # max_epochs=epoch ,profiler="advanced", runs one batch to test that the model works
    # model = DNN(input_size, output_size,config)
    # print("started fitting, model on gpu: ")
    # trainer.fit(model = model)#trainer.fit(model = model,ckpt_path="Data/bi_model_bestrun.ckpt")
    # # trainer.validate(dataloaders=val_dl,model = model,ckpt_path="Bi_experiments/Bilateral_gu_v1.ckpt")
    # print("stopped fitting")
    # trainer.save_checkpoint("Bi_experiments/bi-GU-LPS-V2-Ex2-logmask.ckpt")
    # script = model.to_torchscript()
    # torch.jit.save(script, "Bi_experiments/bi-GU-LPS-V2-Ex2-logmask.pt")





