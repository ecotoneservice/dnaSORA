#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import DataLoader
from statsmodels.nonparametric.smoothers_lowess import lowess
import torch 
import gc
import plotly.graph_objects as go
from torch import nn 
from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor,Compose
from sklearn.model_selection import train_test_split
from datetime import datetime
import re
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pandas as pd
import math
import os
from tqdm import tqdm
from einops import rearrange 
import yaml
import pickle
import matplotlib.pyplot as plt 
from functools import partial, reduce
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import seaborn as sns
import matplotlib

import panel as pn

import holoviews as hv

import random
import sys
import math
import time
import random
from yellowbrick.text import TSNEVisualizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from holoviews import dim, opts
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join('..')))
from src.md import MDDataset, generate_phases_for_dense_validation, MDDenseSet, MDLoadable, MDDense, MDDensePhaseData
from src.models.denoiser import DenoiserModelPipeline, DenoiserModel, get_forward_diffusion_params, forward_add_noise
from src.preprocessing import ChromoDataContainer, ChromoData


device='cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


# Unified Clasifier Attention head

# Positional encoding for attention cls head
class CosinePositionalEncoding(nn.Module):
    def __init__(self, height, width, emb_size):
        super().__init__()
        
        # Calculate sequence length
        seq_len = height * width
        
        # Create position indices
        position = torch.arange(seq_len).float()
        y = position // width
        x = position % width
        
        # Create frequency bands for embedding dimension
        div_term = torch.exp(torch.arange(0, emb_size//2, 2) * (-math.log(10000.0) / (emb_size//2)))
        
        # Calculate positional encodings
        pe = torch.zeros(1, emb_size, seq_len)  # Shape matches your flattened input
        
        # Compute encodings for x and y positions
        pe[:, 0::4, :] = torch.sin(x.unsqueeze(0) * div_term.view(-1, 1))
        pe[:, 1::4, :] = torch.cos(x.unsqueeze(0) * div_term.view(-1, 1))
        pe[:, 2::4, :] = torch.sin(y.unsqueeze(0) * div_term.view(-1, 1))
        pe[:, 3::4, :] = torch.cos(y.unsqueeze(0) * div_term.view(-1, 1))
        
        # Register as buffer (won't be updated during training)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, emb_size, height*width)
        Returns:
            Tensor of same shape with positional encoding added
        """
        B, E, H, W = x.shape
        return x + self.pe.view(1, E, H * W).reshape(1, E, H, W)

# In general similar to DiT block of denoise model (AdaLN but without conditioning shift/scale)
class AttentionBlock(nn.Module):
    def printf(self, *args):
        if self.debug:
            print(*args)
    def __init__(self, emb_size, nhead, dropout=0.01, debug=False):
        
        super().__init__()
        assert emb_size % nhead == 0, "emb_size must be divisible by nhead"
        self.debug = debug
        self.emb_size = emb_size
        self.nhead = nhead
        self.head_dim = emb_size // nhead
        self.dropout = dropout
        
        # Layer norm
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        
        # Multi-head self-attention
        self.wq = nn.Linear(emb_size, emb_size)
        self.wk = nn.Linear(emb_size, emb_size)
        self.wv = nn.Linear(emb_size, emb_size)
        self.lv = nn.Linear(emb_size, emb_size)
        
        
        # Dropout
        self.attn_drop = nn.Dropout(self.dropout)
        self.ff_drop = nn.Dropout(self.dropout)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4 ),
            nn.GELU(),
            nn.Linear(emb_size * 4 , emb_size)
        )
        # print(f"Attention block with {emb_size} embedding size and {nhead} heads created")

    def forward(self, x):
        # Layer norm
        y = self.ln1(x)
        self.printf(f"Input shape: {y.shape}")
        
        # Attention
        q = self.wq(y)
        self.printf(f"Q shape: {q.shape}")
        k = self.wk(y)
        self.printf(f"K shape: {k.shape}")
        v = self.wv(y)
        self.printf(f"V shape: {v.shape}")
        
        # Reshape for multi-head attention
        q = q.view(q.size(0), q.size(1), self.nhead, self.head_dim).permute(0, 2, 1, 3)
        self.printf(f"Q reshaped: {q.shape}")
        k = k.view(k.size(0), k.size(1), self.nhead, self.head_dim).permute(0, 2, 3, 1)
        self.printf(f"K reshaped: {k.shape}")
        v = v.view(v.size(0), v.size(1), self.nhead, self.head_dim).permute(0, 2, 1, 3)
        self.printf(f"V reshaped: {v.shape}")
        
        # Attention computation
        attn = q @ k / math.sqrt(self.emb_size)
        self.printf(f"Attn shape: {attn.shape}")
        attn = torch.softmax(attn, dim=-1)
        self.printf(f"Attn softmax shape: {attn.shape}")
        
        attn = self.attn_drop(attn)
        self.printf(f"Attn dropout shape: {attn.shape}")
        
        y = attn @ v
        self.printf(f"Y shape: {y.shape}")
        
        # Reshape back
        y = y.permute(0, 2, 1, 3)
        self.printf(f"Y reshaped: {y.shape}")
        y = y.reshape(y.size(0), y.size(1), y.size(2)*y.size(3))
        self.printf(f"Y reshaped 2: {y.shape}")
        y = self.lv(y)
        self.printf(f"Y after lv: {y.shape}")
        
        # First residual
        y = x + y
        self.printf(f"Y after residual: {y.shape}")
        
        # Layer norm + FF + residual
        z = self.ln2(y)
        self.printf(f"Z shape: {z.shape}")
        z = self.ff(z)
        self.printf(f"Z after FF: {z.shape}")
        
        z = self.ff_drop(z)
        self.printf(f"Z after FF dropout: {z.shape}")
        
        y = y + z
        self.printf(f"Y after FF residual: {y.shape}")
        return y

# Classification head with attention blocks
class ClassificationHeadAttention(nn.Module):
    def printf(self, *args):
        if self.debug:
            print(*args)
    def __init__(self, channel, width, height, label_num, nhead=8, num_blocks=8, emb_size=8, dropout=0.01, debug=False):
        super().__init__()
        self.debug = debug
        self.channel = channel
        self.width = width
        self.height = height
        self.nhead = nhead
        self.emb_size = emb_size
        self.dropout = dropout
        
        self.block_out = width * height * emb_size
        
        # emb layer
        self.embedding = nn.Sequential(
            nn.Conv2d(channel, emb_size, 1, 1, 0),  # Preserve spatial info
            nn.LayerNorm([emb_size, height, width]),
            nn.GELU(),
            nn.Dropout(self.dropout) 
        )
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.emb_size, nhead, debug=False, dropout=self.dropout)
            for _ in range(num_blocks)
        ])
        # Set debug mode for first block
        self.attention_blocks[0].debug = debug
        
        # Classification head
        self.classifier = nn.Linear(self.block_out, label_num)
        
        # Pos enc
        self.pos_enc = CosinePositionalEncoding(height, width, emb_size)
        
        self.printf(f"Embedding size: {self.emb_size}")
        self.printf(f"Block output size: {self.block_out}")

    def forward(self, x):
        # x shape: (batch, channel, height, width)
        self.printf(f"X input shape: {x.shape}")
        
        x = self.embedding(x)
        self.printf(f"X after embedding: {x.shape}")
        
        x = self.pos_enc(x)
        self.printf(f"X after pos enc: {x.shape}")
        
        x = x.view(x.size(0), -1, self.emb_size)
        self.printf(f"X after view 2: {x.shape}")

        # Process through attention blocks
        for block in self.attention_blocks:
            x = block(x)
            self.printf(f"X after block: {x.shape}")
        
        # Global pooling
        x = x.permute(0, 2, 1)  # (batch, emb_size, seq_len)
        self.printf(f"X before mlp: {x.shape}")
        
        # reshape to 2d
        x = x.reshape(-1, x.size(1) * x.size(2))
        self.printf(f"X before mlp reshape: {x.shape}")
        
        # Classification
        x = self.classifier(x)
        self.printf(f"X after classifier: {x.shape}")
        
        return x
    
if __name__ == '__main__':
    # test 

    # Create random input tensor
    x = torch.randn(2, 1, 28, 28)

    # Create model
    model = ClassificationHeadAttention(channel=1, width=28, height=28, label_num=100, emb_size=8, debug=True)

    # Forward pass
    output = model(x)

    # Check output shape
    print(output.shape)

    del model
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


# dummy CLS head
# used for Zero-shot Classifier, by-passing the classification head
class ClassificationHeadDummy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
# Linear cls head
# Uses to demonstrate the performance of the model with a simple linear head
class ClassificationHeadLinear(nn.Module):
    def __init__(self,channel,width,height,label_num):
        super().__init__()
        self.linear=nn.Linear(channel*width*height,label_num)
        
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x


# In[ ]:


# UNIFIED MODEL
# Consist of pretrained and freezed denoiser model and classifier head
# they are executed in  chain
class ClassifierUnifiedModel(nn.Module):
    def __init__(self, denoiser_model, classifier_head):
        super().__init__()
        self.denoiser_model = denoiser_model
        self.classifier_head = classifier_head
        
    def forward(self, x, t, y, doAblation=False, ablationSlice=None):
        x, denoiserAblation = self.denoiser_model(x, t, y, doAblation, ablationSlice)
        x = self.classifier_head(x)
        return x, denoiserAblation
    
# TEST
if __name__ == '__main__':
    denoiserModel = DenoiserModel(28, 4, 1, 512, 10, 10, 8).cpu()
    classifierHead = ClassificationHeadAttention(1, 28, 28, 100, 8, 8, 8).cpu()
    unifiedModel = ClassifierUnifiedModel(denoiserModel, classifierHead).cpu()
    input = torch.randn(2, 1, 28, 28).cpu()
    t = torch.randint(0, 999, (2,)).cpu()
    y = torch.randint(0, 10, (2,)).cpu()
    output, denoiserAblation = unifiedModel(input, t, y, True, 2)
    print(f"Output shape: {output.shape}")
    print(f"Denoiser ablation len: {len(denoiserAblation)}")


# In[ ]:


# DATACLASSES

# Model parameters for attention head
@dataclass
class ModelParametersAttn:
    channel: int
    width: int
    height: int
    nhead: int
    num_blocks: int
    emb_size: int
    dropout: float
    label_num: int
    
# Model parameters for linear head
@dataclass
class ModelParametersLinear:
    channel: int
    width: int
    height: int
    label_num: int
    
# Dummy head model parameters
@dataclass
class ModelParametersDummy:
    pass
    
# Mapping for model parameters and classes
header_mapping_params = {
    'attn': ModelParametersAttn,
    'linear': ModelParametersLinear,
    'dummy': ModelParametersDummy
}

header_mapping_classes = {
    'attn': ClassificationHeadAttention,
    'linear': ClassificationHeadLinear,
    'dummy': ClassificationHeadDummy
}

# Unified Model config
@dataclass
class Model:
    description: str
    denoiser_model: str
    cls_head: str
    parameters: object

# Unified Model training config
@dataclass
class Training:
    skip: bool
    n_epochs: int
    validation_per_epoch: int
    validation_portion: float
    batch_size: int
    learning_rate: float
    dataset: str
    label_num: int
    timestep: int
    label_count: int
    

# Unified Model top level config
@dataclass
class ClassifierModelConfig:
    model: Model
    training: Training
    
# The idea of this pipeline is to simplify the process of loading and saving the model and its dataset, using only config file name
# training and inference going outside of this class, maybe good idea to move inference here later, similar to StableDiffusionPipeline
class ClassifierModelPipeline:
    model = None # unified model
    dataset = None # training dataset
    config = None # config which loaded from yaml
    file_name = None # reference name of model config, same as file name for distinguishing
    denoiser_model = None # denoiser model
    
    def load_dataset(self):
        # assuming only MD datasets
        print(f"Loading dataset {self.config.training.dataset}")
        self.dataset = MDDataset.load(self.config.training.dataset, dataset_cls=MDDataset)
        print(f"Dataset loaded")
        
    def load(
            config_name, 
            config_folder = '../configs/classifier', 
            config_folder_denoiser = '../configs/denoiser',
            checkpoints_folder='../checkpoints/classifier', 
            skip_data_load=False
        ):
        name = config_name.replace('.yaml', '') 
        full_path_config = os.path.join(config_folder, f"{name}.yaml")
        full_path_data = os.path.join(checkpoints_folder, f"{name}.pth")
        data = None
        model = None
        denoiser_model = None
        
        with open(full_path_config, 'r') as file:
            print(f"Loading config from {full_path_config}")
            config = yaml.load(file, Loader=yaml.FullLoader)
            
            config = ClassifierModelConfig(**config)
            config.model = Model(**config.model)
            head = config.model.cls_head
            print(f"Loading model with head {head}")
            head_param_class = header_mapping_params[head]
            head_class = header_mapping_classes[head]
            config.model.parameters = head_param_class(**config.model.parameters)
            config.training = Training(**config.training)
            
            print(config)
            print(f"Loading denoiser model {config.model.denoiser_model}")
            denoiser_model = DenoiserModelPipeline.load(config.model.denoiser_model, config_folder_denoiser, skip_data_load=False)
            print(f"Denoiser model loaded")
            print(f"Creating classifier model")
            classifier_model = head_class(**config.model.parameters.__dict__)
            print(f"Classifier model created")
            print(f"Creating unified model")
            model = ClassifierUnifiedModel(denoiser_model.model, classifier_model)
            print(f"Unified model created")
            
        if not skip_data_load:
            print(f"Loading data from {full_path_data}")
            data = torch.load(full_path_data)
            model.load_state_dict(data)
            
        pipeline = DenoiserModelPipeline()
        pipeline.model = model
        pipeline.denoiser_model = denoiser_model
        pipeline.config = config
        pipeline.file_name = name
        return pipeline


# In[ ]:


# predict function for linear and attention heads
def predictCLS(model, image, alphas_cumprod, TIMESTEP, LABEL_NUM, LABEL_COUNT):
    t = torch.full((image.shape[0],),TIMESTEP,dtype=torch.long).to(device)
    y = torch.full((image.shape[0],),LABEL_NUM,dtype=torch.long).to(device)
    x = image*2-1
    x_t,noise = forward_add_noise(x,t, alphas_cumprod)
    labels_pred,_ = model(x_t,t,y)
    max_idx = torch.argmax(labels_pred, dim=1).detach().cpu().numpy()
    return max_idx

# predict function for dummy head
# Implementation of Zero-shot Classifier
def predictDenoise(model, image, alphas_cumprod, TIMESTEP, LABEL_NUM, LABEL_COUNT, show_progress=False):
    batch_size = image.shape[0]
    t = torch.full((batch_size,), TIMESTEP, dtype=torch.long).to(device)
    x = image * 2 - 1
    noise_diffs = []
    for i in tqdm(range(LABEL_COUNT), disable=not show_progress):
        x_t, noise = forward_add_noise(x, t, alphas_cumprod)
        y = torch.full((batch_size,), i, dtype=torch.long).to(device)
        noise_pred, _ = model(x_t, t, y)
        noise_diff = (noise-noise_pred) ** 2 # calculate difference MSE
        noise_diffs.append(noise_diff.detach().cpu().numpy())
    noise_diffs = np.array(noise_diffs)
    noise_diffs = noise_diffs.transpose(1, 0, 2, 3, 4)
    noise_diffs = noise_diffs.reshape(batch_size, LABEL_COUNT, -1)
    noise_diffs = np.sum(noise_diffs, axis=2) # sum together pixels to reduce dims
    final_scores = np.argmin(noise_diffs, axis=-1)
    return final_scores

# for training evaluation
# function which evaluate model on test set, by using predict function
# it calculates accuracy and error percentage
# it also calculates error percentage as absolute difference between predicted and real value
# for attn and linear heads predFN is predictCLS, for dummy head predFN is predictDenoise
def evaluate_test_set(model, predFN, loader, alphas_cumprod, TIMESTEP, LABEL_NUM, CLS_CNT, iterations):
    correct = 0
    error_percentage = 0    
    pred = None
    label_values = None
    print(f"Iterations: {iterations}")
    for i in tqdm(range(iterations)):
        datachunk = next(iter(loader))
        gxx = datachunk['gxx']
        labels_classifier = datachunk['labels_classifier']
        gxx = gxx.to(device)
        labels_classifier = labels_classifier.to(device)
        # print(f"Img shape: {img.shape}, label shape: {label.shape}")
        pred_result = predFN(model, gxx, alphas_cumprod, TIMESTEP, LABEL_NUM, CLS_CNT)
        label_values_result = torch.argmax(labels_classifier, dim=1).cpu().numpy()
        if pred is None:
            pred = pred_result
            label_values = label_values_result
        else:
            pred = np.concatenate((pred, pred_result))
            label_values = np.concatenate((label_values, label_values_result))
    to_test = len(label_values)
    for i in range(len(pred)):
        if pred[i] == label_values[i]:
            correct += 1
        error_percentage = error_percentage + math.fabs(pred[i] - label_values[i]) / CLS_CNT
        
    error_percentage = error_percentage * 100 / to_test
            
    print(f"Accuracy: {correct/to_test}, correct: {correct}, total: {to_test}, error percentage: {error_percentage}")
    result = {
        'accuracy': correct/to_test,
        'correct': correct,
        'total': to_test,
        'error_percentage': error_percentage
    }
    return result
    
# for evaluation of model against dense MD
# it calculates accuracy and error percentage
# it also calculates error percentage as absolute difference between predicted and real value
# for attn and linear heads predFN is predictCLS, for dummy head predFN is predictDenoise 
def evaluate_dense_chromosomes(model, predFN, val_dataset: MDDenseSet, alphas_cumprod, TIMESTEPS, LABEL_NUM, CLS_CNT, show_details=False, BATCH_SIZE=1, window_len=28*28):
    # correction parameter, should be close to 1
    pred_to_section_a = CLS_CNT / window_len
    print(f"Pred to section: {pred_to_section_a}")
    # run through all gausians phases and calculate accuracy
    correct = 0
    accuracy = 0
    min_diff = 0
    max_diff = 0
    total_chromo = 0
    total_error = 0
    total_abs_percentage_error = 0
    preds = []
    pairs_measured_calculated = []
    for mdDense in tqdm(val_dataset):
        
        phase_data = mdDense.phaseData
        mdPhasesDataset = mdDense.data
        chromoData = mdDense.chromo
        max_idx = mdDense.max_idx
        
        g_number = chromoData.g_number
        array = chromoData.array
        chrom_label = chromoData.chrom_label
        
        
        correct_chromo = 0
        accuracy_chromo = 0
        min_diff_chromo = 10**10
        max_diff_chromo = 0
        pred_avg = 0
        preds_chromo = []
        
        phases_count = len(phase_data)
        
        chromoDataLoader = DataLoader(mdPhasesDataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # split to batches
        indicies_pred = []
        indicies_val = []
        # print (f"Phases batch split: {len(phases_batch_split)}, original len: {len(phases_batch)}")
        for datachunk in chromoDataLoader:
            
            gxx = datachunk['gxx'].to(device)
            labels_classifier = datachunk['labels_classifier'].numpy().copy()
            labels_classifier = np.argmax(labels_classifier, axis=1)
            
            # print(f"gxx shape: {gxx.shape}")
            # print(f"Labels classifier: {labels_classifier.shape}")
            
            indicies_pred_val = predFN(model, gxx, alphas_cumprod, TIMESTEPS, LABEL_NUM, CLS_CNT)
            
            # print(f"Indicies pred val: {indicies_pred_val.shape}")
            
            indicies_pred = np.concatenate((indicies_pred, indicies_pred_val.astype(int)))
            indicies_val = np.concatenate((indicies_val, labels_classifier.astype(int)))
            
            # print(f"Indicies pred: {indicies_pred.shape}")
            # print(f"Indicies val: {indicies_val.shape}")
            

        for i in range(len(indicies_pred)):
            
            mdDensePhaseData = phase_data[i]
            mu_recalculated = mdDensePhaseData.mu_recalculated
            indicies = mdDensePhaseData.indicies
            slice_size = len(indicies)
            
            if indicies_pred[i] == indicies_val[i]:
                correct_chromo += 1
            
            # print(f"Indicies {indicies}")
            # print(f"Indicies pred: {indicies_pred}, Indicies val: {indicies_val}")
            # print(f"Indicies pred: {indicies_pred[i]}, Indicies val: {indicies_val[i]}")
            pred_val = indicies[int(indicies_pred[i])]
            pred_avg += pred_val
            preds_chromo.append(pred_val)
            
            if mu_recalculated > 0:
                accuracy_curr = math.fabs(pred_val - mu_recalculated) / mu_recalculated
            else:
                accuracy_curr = 0
            accuracy_chromo += accuracy_curr * slice_size
            
            diff = math.fabs(pred_val - mu_recalculated)
            min_diff_chromo = min(min_diff_chromo, diff)
            max_diff_chromo = max(max_diff_chromo, diff)
            preds.append(indicies_pred[i])
    
        accuracy_chromo = accuracy_chromo / phases_count
        
        pred_calculated_from_avg = pred_avg / phases_count
        error_calc = np.fabs(pred_calculated_from_avg - max_idx)
        
        # calculate absolute percentage error as absolute error / len of array. Lower = better
        abs_percentage_error = math.fabs(pred_calculated_from_avg - max_idx) / len(array) * 100
        total_abs_percentage_error += abs_percentage_error
        
        total_error += error_calc
        
        if show_details:    
            print(f"========================================================================")
            print(f"G_number: {g_number}, Chrom: {chrom_label}, mu: {max_idx}")
            print(f"Correct: {correct_chromo}, Total: {phases_count}, Precision: {accuracy_chromo}, average pred: {pred_calculated_from_avg}, max_idx: {max_idx}, error absolute: {error_calc}, abs percentage error: {abs_percentage_error}")
            print(f"Mu measured {array['POS'].iloc[max_idx]}, Mu calculated: {array['POS'].iloc[int(pred_calculated_from_avg)]}, error: {error_calc}, abs percentage error: {abs_percentage_error}")
            
        pairs_measured_calculated.append((g_number, chrom_label, array['POS'].iloc[max_idx], array['POS'].iloc[int(pred_calculated_from_avg)]))
        accuracy += accuracy_chromo
        correct += correct_chromo
        total_chromo+=phases_count
            
    accuracy = accuracy / len(val_dataset)
    total_error = total_error / len(val_dataset)
    total_abs_percentage_error = total_abs_percentage_error / len(val_dataset)
    print("=======TOTAL=======")
    print(f"Correct: {correct}, Total: {total_chromo}, Accuracy% { correct*100/total_chromo }, Precision : {accuracy}", f"Total error abs: {total_error}", f"Total abs percentage error: {total_abs_percentage_error}")

    
    result = {
        'correct': correct,
        'total': total_chromo,
        'accuracy': correct*100/total_chromo,
        'precision': accuracy,
        'total_error': total_error,
        'total_abs_percentage_error': total_abs_percentage_error,
        'pairs_measured_calculated': pairs_measured_calculated,
        'preds': preds,
        'pred_avg': pred_calculated_from_avg
    }
    return result

# for evaluation of model against dense MD sets
# convenience method, given we have 4 sets, but in future it can be generalized
def evaluate_unified_classifier_model(
    model,
    distilled_val_sets,
    alphas_cumprod,
    timestep,
    label_num,
    label_count,
    batch_size,
    predictFn): # TODO: refactor this to have better generalization
    distiled_real_mu, distiled_real_lowess, distiled_generated_mu, distiled_generated_lowess = distilled_val_sets # TODO: refactor this to have better generalization
    # TODO: refactor this to have better generalization
    # validation on RD
                
    print("Validation on Dense RD Mu")
    result_rel_mu = evaluate_dense_chromosomes(model, predictFn, distiled_real_mu, alphas_cumprod, timestep, label_num, label_count, show_details=True, BATCH_SIZE=batch_size)
    print("Validation on Dense RD Lowess")
    result_rel_lowess = evaluate_dense_chromosomes(model, predictFn,distiled_real_lowess, alphas_cumprod, timestep, label_num, label_count, BATCH_SIZE=batch_size)
    print("Validation on Dense MD Mu")
    result_gen_mu = evaluate_dense_chromosomes(model, predictFn, distiled_generated_mu, alphas_cumprod, timestep, label_num, label_count, BATCH_SIZE=batch_size)   
    print("Validation on Dense MD Lowess")
    result_gen_lowess = evaluate_dense_chromosomes(model, predictFn, distiled_generated_lowess, alphas_cumprod, timestep, label_num, label_count, BATCH_SIZE=batch_size)
    return result_rel_mu, result_rel_lowess, result_gen_mu, result_gen_lowess

