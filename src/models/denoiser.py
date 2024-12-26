#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import math
import panel as pn
import yaml
import holoviews as hv
import altair as alt
alt.data_transformers.disable_max_rows()
hv.extension("plotly")
pn.extension("plotly")
pn.config.theme = 'dark'
hv.renderer('plotly').theme = 'dark'
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import torch
import yaml
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import os
from tqdm import tqdm
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join('..')))
from src.md import MDDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# Modeling

# Get forward diffusion parameters.
# In general, creates noise scheduler which implies that noise would be added to the image in each step uniformly cumulatively.
# T: Number of steps
def get_forward_diffusion_params(T=1000):
    # Forward diffusion calculation parameters
    betas = torch.linspace(0.0001, 0.02, T)  # (T,)
    alphas = 1 - betas  # (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=-1)  # Cumulative product of alpha_t (T,) [a1, a2, a3, ....] -> [a1, a1*a2, a1*a2*a3, .....]
    alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)  # Cumulative product of alpha_t-1 (T,), [1, a1, a1*a2, a1*a2*a3, .....]
    variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # Variance used for denoising (T,)
    return T, betas, alphas, alphas_cumprod, alphas_cumprod_prev, variance

# Time embedding module for DiT denoiser
class TimeEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_emb_size=emb_size//2
        half_emb=torch.exp(torch.arange(self.half_emb_size)*(-1*math.log(10000)/(self.half_emb_size-1)))
        self.register_buffer('half_emb',half_emb)

    def forward(self,t):
        t=t.view(t.size(0),1)
        half_emb=self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size)
        half_emb_t=half_emb*t
        embs_t=torch.cat((half_emb_t.sin(),half_emb_t.cos()),dim=-1)
        return embs_t
    
# DiT block for DiT denoiser
# Implementation of ADA-LN (Adaptive Layer Normalization) for DiT denoiser
class DiTBlock(nn.Module):
    def __init__(self,emb_size,nhead):
        super().__init__()
        
        self.emb_size=emb_size
        self.nhead=nhead
        
        # conditioning
        self.alpha1=nn.Linear(emb_size,emb_size)
        self.beta1=nn.Linear(emb_size,emb_size)        
        self.gamma1=nn.Linear(emb_size,emb_size)
        self.gamma2=nn.Linear(emb_size,emb_size)
        self.alpha2=nn.Linear(emb_size,emb_size)
        self.beta2=nn.Linear(emb_size,emb_size)
        
        # layer norm
        self.ln1=nn.LayerNorm(emb_size)
        self.ln2=nn.LayerNorm(emb_size)
        
        # multi-head self-attention
        self.wq=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.wk=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.wv=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.lv=nn.Linear(nhead*emb_size,emb_size)
        
        # feed-forward
        self.ff=nn.Sequential(
            nn.Linear(emb_size,emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4,emb_size)
        )

    def forward(self,x,cond, doAblation = False, ablation_slice=None):   # x:(batch,seq_len,emb_size), cond:(batch,emb_size)
        # conditioning (batch,emb_size)
        
        gamma1_val=self.gamma1(cond)
        beta1_val=self.beta1(cond)
        alpha1_val=self.alpha1(cond)
        
        gamma2_val=self.gamma2(cond)
        beta2_val=self.beta2(cond)
        alpha2_val=self.alpha2(cond)
        
        # layer norm
        y=self.ln1(x) # (batch,seq_len,emb_size)
        
        # scale&shift
        y=y*(1+gamma1_val.unsqueeze(1))+beta1_val.unsqueeze(1) 
        
        # attention
        q=self.wq(y)    # (batch,seq_len,nhead*emb_size)
        k=self.wk(y)    # (batch,seq_len,nhead*emb_size)    
        v=self.wv(y)    # (batch,seq_len,nhead*emb_size)
        q=q.view(q.size(0),q.size(1),self.nhead,self.emb_size).permute(0,2,1,3) # (batch,nhead,seq_len,emb_size)
        k=k.view(k.size(0),k.size(1),self.nhead,self.emb_size).permute(0,2,3,1) # (batch,nhead,emb_size,seq_len)
        v=v.view(v.size(0),v.size(1),self.nhead,self.emb_size).permute(0,2,1,3) # (batch,nhead,seq_len,emb_size)
        attn=q@k/math.sqrt(q.size(2))   # (batch,nhead,seq_len,seq_len)
        attn=torch.softmax(attn,dim=-1)   # (batch,nhead,seq_len,seq_len)
        y=attn@v    # (batch,nhead,seq_len,emb_size)
        y=y.permute(0,2,1,3) # (batch,seq_len,nhead,emb_size)
        y=y.reshape(y.size(0),y.size(1),y.size(2)*y.size(3))    # (batch,seq_len,nhead*emb_size)
        y=self.lv(y)    # (batch,seq_len,emb_size)
        
        # scale
        y=y*alpha1_val.unsqueeze(1)
        # redisual
        y=x+y  
        
        # layer norm
        z=self.ln2(y)
        # scale&shift
        z=z*(1+gamma2_val.unsqueeze(1))+beta2_val.unsqueeze(1)
        # feef-forward
        z=self.ff(z)
        # scale 
        z=z*alpha2_val.unsqueeze(1)
        # residual
        
        ablation = None
        if doAblation:
            ablation = {
                'q': q[:ablation_slice].clone().detach().cpu().numpy(),
                'k': k[:ablation_slice].clone().detach().cpu().numpy(),
                'v': v[:ablation_slice].clone().detach().cpu().numpy(),
                'attn': attn[:ablation_slice].clone().detach().cpu().numpy(),
            }
        
        return y+z, ablation
    
# DiT denoiser model
class DenoiserModel(nn.Module):
    def __init__(self,img_size,patch_size,channel,emb_size,label_num,dit_num,head):
        super().__init__()
        
        self.patch_size=patch_size
        self.patch_count=img_size//self.patch_size
        self.channel=channel
        
        # patchify
        self.conv=nn.Conv2d(in_channels=channel,out_channels=channel*patch_size**2,kernel_size=patch_size,padding=0,stride=patch_size) 
        self.patch_emb=nn.Linear(in_features=channel*patch_size**2,out_features=emb_size) 
        self.patch_pos_emb=nn.Parameter(torch.rand(1,self.patch_count**2,emb_size))
        
        # time emb
        self.time_emb=nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size)
        )

        # label emb
        self.label_emb=nn.Embedding(num_embeddings=label_num,embedding_dim=emb_size)
        
        # DiT Blocks
        self.dits=nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size,head))
        
        # layer norm
        self.ln=nn.LayerNorm(emb_size)
        
        # linear back to patch
        self.linear=nn.Linear(emb_size,channel*patch_size**2)

    def forward(self,x,t,y, doAblation=False, ablation_slice=None): # x:(batch,channel,height,width)   t:(batch,)  y:(batch,)
        
        ablation = None
        if doAblation:
            if ablation_slice is None:
                ablation_slice = t.shape[0]
            ablation = {
                't': t[:ablation_slice].clone().detach().cpu().numpy(),
                'y': y[:ablation_slice].clone().detach().cpu().numpy(),
                'DIT': []
            }
        
        # label emb
        y_emb=self.label_emb(y) #   (batch,emb_size)
        # time emb
        t_emb=self.time_emb(t)  #   (batch,emb_size)
        
        # condition emb
        cond=y_emb+t_emb
        
        # patch emb
        x=self.conv(x)  # (batch,new_channel,patch_count,patch_count)
        x=x.permute(0,2,3,1)    # (batch,patch_count,patch_count,new_channel)
        x=x.view(x.size(0),self.patch_count*self.patch_count,x.size(3)) # (batch,patch_count**2,new_channel)
        
        x=self.patch_emb(x) # (batch,patch_count**2,emb_size)
        x=x+self.patch_pos_emb # (batch,patch_count**2,emb_size)
        
        # dit blocks
        for dit in self.dits:
            x, ablation_sample=dit(x,cond, doAblation, ablation_slice)
            if doAblation:
                ablation['DIT'].append(ablation_sample)
        
        # # layer norm
        x=self.ln(x)    #   (batch,patch_count**2,emb_size)
        
        # # linear back to patch
        x=self.linear(x)    # (batch,patch_count**2,channel*patch_size*patch_size)
        
        # reshape
        x=x.view(x.size(0),self.patch_count,self.patch_count,self.channel,self.patch_size,self.patch_size)  # (batch,patch_count,patch_count,channel,patch_size,patch_size)
        x=x.permute(0,3,1,2,4,5)    # (batch,channel,patch_count(H),patch_count(W),patch_size(H),patch_size(W))
        x=x.permute(0,1,2,4,3,5)    # (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x=x.reshape(x.size(0),self.channel,self.patch_count*self.patch_size,self.patch_count*self.patch_size)   # (batch,channel,img_size,img_size)
        return x, ablation

# Execute forward noise addition
def forward_add_noise(x, t, alphas_cumprod):  # batch_x: (batch, channel, height, width), batch_t: (batch_size,)
    noise = torch.randn_like(x)  # Generate Gaussian noise for each image at step t (batch, channel, height, width)
    batch_alphas_cumprod = alphas_cumprod[t].view(x.size(0), 1, 1, 1)
    x = torch.sqrt(batch_alphas_cumprod) * x + torch.sqrt(1 - batch_alphas_cumprod) * noise  # Generate noisy image at step t based on the formula
    return x, noise

# Trainer for DiT denoiser
def train_diffusion_model(dataset,
                          model,
                          run_name = 'unnamed_denoiser_model',
                          n_epochs = 100,
                          batch_size = 32,
                          lr=10e-3,
                          ablation_in_epoch_per_each_epochs = 0,
                          validation_per_epoch = 0,
                          validation_portion = 0.1,
                          ablation_batch_slice = 2, # number of samples to slice for ablation
                          checkpoints_folder='../checkpoints/denoiser',
                          ablation_folder='../ablation/denoiser'):
    # Print model architecture size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-----------------------------")
    timestamp_int = int(datetime.now().timestamp())
    log_name = f"{timestamp_int}_{run_name}"
    print(f"Log name: {log_name}")
    writer = SummaryWriter(log_dir=f'../logs/{log_name}')
    
    # create dirs for ablation_file_checkpoint and path_checkpoint root files
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(ablation_folder, exist_ok=True)
    
    # setting ablation and checkpoint file paths
    path_checkpoint = os.path.join(checkpoints_folder, f"{run_name}.pth")
    ablation_file_checkpoint = os.path.join(ablation_folder, f"{run_name}.pkl")
    print(f"Checkpoint file: {path_checkpoint}")
    print(f"Ablation file: {ablation_file_checkpoint}")
    
    # create train and val dataset
    train_ds, val_ds = dataset.split(test_size=validation_portion)


    optimzer = torch.optim.Adam(model.parameters(), lr=lr)  # Optimizer
    loss_fn = nn.L1Loss()  # Loss function (Mean Absolute Error)
    tqdm_epoch = trange(n_epochs)
    current_step = 0
    
    ablation_samples_epoch = []

    model.to(device)
    model.train()
    
    dataLoader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataLoader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    T, betas, alphas, alphas_cumprod, alphas_cumprod_prev, variance = get_forward_diffusion_params()
    
    for epoch in tqdm(tqdm_epoch):
        ablation_samples = []
        epoch_iter = 0
        
        
        if validation_per_epoch > 0 and epoch % validation_per_epoch == 0:
            if epoch > 0:
                torch.save(model.state_dict(),path_checkpoint)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for datachunk in val_dataLoader:
                    gxx = datachunk['gxx']
                    labels_values = datachunk['labels_values']

                    # rescale from [0,1] to [-1,1]
                    x = gxx * 2 - 1 
                    
                    # get random time step
                    t = torch.randint(0, T, (gxx.size(0),))

                    # alphas_cumprod schedules noise addition
                    x,noise=forward_add_noise(x,t, alphas_cumprod)
                    
                    pred_noise, _ = model(x.to(device),t.to(device),labels_values.to(device))
                    
                    loss=loss_fn(pred_noise,noise.to(device))
                    
                    val_loss += loss.item()
                    
                    break # change for more samples to validate
                
            writer.add_scalar('Loss/val', val_loss, current_step)
        
        model.train()
        
        for datachunk in dataLoader:
            perform_ablation = ablation_in_epoch_per_each_epochs > 0 and epoch_iter % ablation_in_epoch_per_each_epochs == 0
            
            gxx = datachunk['gxx']
            labels_values = datachunk['labels_values']
            
            x = gxx * 2 - 1 
            t = torch.randint(0, T, (gxx.size(0),))
            
            x,noise=forward_add_noise(x,t, alphas_cumprod) # x: noisy image, noise: added noise
            
            x = x.to(device)
            t = t.to(device)
            y = labels_values.to(device)
            
            pred_noise, ablation=model(x,t,y, perform_ablation, ablation_batch_slice)
            
            if perform_ablation:
                ablation['epoch'] = epoch
                ablation['iter'] = epoch_iter
                ablation_samples.append(ablation)

            loss=loss_fn(pred_noise,noise.to(device))
            
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            
            current_step = current_step + 1
            epoch_iter = epoch_iter + 1
            
            if current_step % 10 == 0:
                writer.add_scalar('Loss/train', loss, current_step)
                
        if len(ablation_samples) > 0:
            ablation_samples_epoch.append(ablation_samples)    
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), path_checkpoint)
    torch.save(model.state_dict(), path_checkpoint)
    print(f"Model saved to {path_checkpoint}")
    if ablation_file_checkpoint and len(ablation_samples_epoch) > 0:
        with open(ablation_file_checkpoint, 'wb') as f:
            pickle.dump(ablation_samples_epoch, f)
        print(f"Ablation samples saved to {ablation_file_checkpoint}")
    else:
        print(f"No ablation samples to save")
        
# Metadata

# Model parameters
@dataclass
class ModelParameters:
    img_size: int
    patch_size: int
    channel: int
    emb_size: int
    label_num: int
    dit_num: int
    head: int

# Model configuration
@dataclass
class Model:
    description: str
    parameters: ModelParameters

# Training configuration
@dataclass
class Training:
    skip: bool
    n_epochs: int
    learning_rate: float
    dataset: str
    batch_size: int
    validation_per_epoch: int
    validation_portion: float
    ablation_in_epoch_per_each_epochs: int
    ablation_batch_slice: int

# Denoiser model configuration
@dataclass
class DenoiserModelConfig:
    model: Model
    training: Training
    
# The idea of this pipeline is to simplify the process of loading and saving the model and its dataset, using only config file name
# training and inference going outside of this class, maybe good idea to move inference here later, similar to StableDiffusionPipeline
class DenoiserModelPipeline:
    
    #DiT transformer model
    model = None # torch model
    dataset = None # training dataset
    config = None # config which loaded from yaml
    file_name = None # reference name of model config, same as file name for distinguishing
    
    def load_dataset(self):
        # assuming only MD datasets
        print(f"Loading dataset {self.config.training.dataset}")
        self.dataset = MDDataset.load(self.config.training.dataset, dataset_cls=MDDataset)
        print(f"Dataset loaded")
    
    def load(config_name, config_folder = '../configs/denoiser', checkpoints_folder='../checkpoints/denoiser', skip_data_load=False):
        name = config_name.replace('.yaml', '') 
        full_path_config = os.path.join(config_folder, f"{name}.yaml")
        full_path_data = os.path.join(checkpoints_folder, f"{name}.pth")
        data = None
        model = None
        
        with open(full_path_config, 'r') as file:
            print(f"Loading config from {full_path_config}")
            config = yaml.load(file, Loader=yaml.FullLoader)
            
            config = DenoiserModelConfig(**config)
            config.model = Model(**config.model)
            config.model.parameters = ModelParameters(**config.model.parameters)
            config.training = Training(**config.training)
            
            print(config)
            print(f"Creating model {config.model.description}")
            model = DenoiserModel(
                img_size=config.model.parameters.img_size,
                patch_size=config.model.parameters.patch_size,
                channel=config.model.parameters.channel,
                emb_size=config.model.parameters.emb_size,
                label_num=config.model.parameters.label_num,
                dit_num=config.model.parameters.dit_num,
                head=config.model.parameters.head
            )
        
        if not skip_data_load:
            print(f"Loading data from {full_path_data}")
            if os.path.exists(full_path_data):
                data = torch.load(full_path_data)
                model.load_state_dict(data)
            else:
                print(f"Data not found in {full_path_data}")
            
        pipeline = DenoiserModelPipeline()
        pipeline.model = model
        pipeline.config = config
        pipeline.file_name = name
        return pipeline
        
if __name__ == "__main__":
    # testing the model
    model = DenoiserModel(64, 4, 3, 512, 10, 10, 8)
    model.cpu()
    input = torch.randn(2, 3, 64, 64).cpu()
    t = torch.randint(0, 1000, (2,)).cpu()
    y = torch.randint(0, 10, (2,)).cpu()
    output, ablation = model(input, t, y, True, 2)
    print(output.shape)
    
  

