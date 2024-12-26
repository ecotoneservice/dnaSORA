#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

import io
import pickle
import numpy as np
import math
import panel as pn
from PIL import Image
import holoviews as hv
import altair as alt
from sklearn.decomposition import PCA
alt.data_transformers.disable_max_rows()
hv.extension("plotly")
pn.extension("plotly")
pn.config.theme = 'dark'
hv.renderer('plotly').theme = 'dark'
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
from tqdm import tqdm, trange
import lpips
import lpips
import os
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor,Compose
import torchvision
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
from src.models.unified_classifier import ClassifierModelPipeline, evaluate_test_set, evaluate_unified_classifier_model, predictDenoise, predictCLS, ClassificationHeadDummy
from src.md import MDDataset, generate_phases_for_dense_validation, MDDenseSet, MDLoadable, MDDense, MDDensePhaseData
from src.models.denoiser import get_forward_diffusion_params, forward_add_noise
from src.preprocessing import ChromoDataContainer, ChromoData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# Separate classifier model, for evaluation against unified one
# This is a simple convolutional classifier model, uses the same training data as the unified model

class ConvChromoClassifier(nn.Module):
    def printf(self, *args):
        if self.debug:
            print(*args)

    def __init__(self, seq_len, class_len, debug=False, dropout_val=0.5):
        super(ConvChromoClassifier, self).__init__()
        self.debug = debug
        self.dropout_val = dropout_val
        self.input_dim = int(math.sqrt(seq_len))
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Calculate the size of the flattened features
        self.flat_features = 256 * (self.input_dim // 8) * (self.input_dim // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, class_len)
        
        # Dropout layers
        self.dropout = nn.Dropout(self.dropout_val)
    
    def forward(self, x):
        self.printf(f"Input shape: {x.shape}")
        
        # Convolutional layers with ReLU, batch norm, max pooling, and dropout
        x = F.relu(self.bn1(self.conv1(x)))
        self.printf(f"After conv1: {x.shape}")
        x = F.max_pool2d(x, 2)
        self.printf(f"After max_pool1: {x.shape}")
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        self.printf(f"After conv2: {x.shape}")
        x = F.max_pool2d(x, 2)
        self.printf(f"After max_pool2: {x.shape}")
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        self.printf(f"After conv3: {x.shape}")
        x = F.max_pool2d(x, 2)
        self.printf(f"After max_pool3: {x.shape}")
        x = self.dropout(x)
        
        # Flatten the features
        x = x.view(-1, self.flat_features)
        self.printf(f"After flattening: {x.shape}")
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        self.printf(f"After fc1: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        self.printf(f"After fc2: {x.shape}")
        x = self.dropout(x)
        x = self.fc3(x)
        self.printf(f"Output shape: {x.shape}")
        
        # x = torch.softmax(x, dim=1)
        
        return x

# Predict function for the ConvChromoClassifier model
# Created for convenient compatibility with general evaluator function
def ConvPredictCLS(model, image, alphas_cumprod, TIMESTEP, LABEL_NUM, LABEL_COUNT):
    # rescale from [0, 1] to [-1, 1]
    x = image*2-1
    labels_pred = model(x)
    max_idx = torch.argmax(labels_pred, dim=1).detach().cpu().numpy()
    return max_idx

# Training function for the ConvChromoClassifier model
#TODO: refactor, unify trainers to single general trainer
def train_conv_classifier_model(
    dataset,
    model,
    run_name = 'unnamed_unified_classifier_model',
    n_epochs = 100,
    batch_size = 32,
    
    timestep = 1, # fixed timestep for the base denoiser model
    label_num = 5, # fixed label num for the base denoiser model
    label_count = 10, # fixed label count for the base denoiser model
    
    lr=10e-3,
    validation_portion=0.1,
    validation_per_epoch = 0,
    validation_samples = 1024,
    
    distilled_val_sets = None, # TODO: refactor this to have better generalization
    
    checkpoints_folder='../checkpoints/classifier'
):
    
    model = model.to(device)
    
    model.train()

    # loss for classification
    loss_fn=nn.CrossEntropyLoss()

    # optimizer for classification head only
    optimzer=torch.optim.AdamW(
        model.parameters(),
        lr=lr)
    
    # define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimzer, step_size=10, gamma=0.9)
    train_ds, val_ds = dataset.split(test_size=validation_portion)
    
    # creating directory for checkpoints
    os.makedirs(checkpoints_folder, exist_ok=True)
    
    # setting checkpoint paths
    path_checkpoint = os.path.join(checkpoints_folder, f"{run_name}.pth")


    dataloader=DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last = True,
        num_workers=10,
        persistent_workers=True
    )
    
    dataloader_eval = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last = True,
        num_workers=10,
        persistent_workers=True
    )

    model.train()
    timestamp_int = int(datetime.now().timestamp())
    log_name = f"{timestamp_int}_{run_name}"
    print(f"Log name: {log_name}")
    writer = SummaryWriter(log_dir=f'../logs/{log_name}')
    
    alphas_cumprod = None
    timestep = None
    label_num = None

    iter_count=0
    for epoch in tqdm(range(n_epochs)):
        epoch_iter = 0
        
        if epoch % validation_per_epoch == 0:
            if epoch > 0:
                torch.save(model.state_dict(),path_checkpoint)
                print("Model saved")
            print("Validating")
            
            # VALIDATION
            
            model.eval()
            
            with torch.no_grad():
                
                # validation on random samples from validation set
                
                print("Validation on val samples")
                result_val = evaluate_test_set(model, ConvPredictCLS, dataloader_eval, alphas_cumprod, timestep, label_num, label_count, iterations=validation_samples // batch_size)
                writer.add_scalar('Error/val', result_val['error_percentage'], iter_count)
                
                
                # TODO: refactor this to have better generalization
                # validation on RD
                
                result_rel_mu, result_rel_lowess, result_gen_mu, result_gen_lowess = evaluate_unified_classifier_model(
                    model,
                    distilled_val_sets,
                    alphas_cumprod,
                    timestep,
                    label_num,
                    label_count,
                    batch_size,
                    ConvPredictCLS  
                )
                
                writer.add_scalar('Error/real_mu', result_rel_mu['total_abs_percentage_error'], iter_count)
                writer.add_scalar('Error/real_lowess', result_rel_lowess['total_abs_percentage_error'], iter_count)
                writer.add_scalar('Error/gen_mu', result_gen_mu['total_abs_percentage_error'], iter_count)
                writer.add_scalar('Error/gen_lowess', result_gen_lowess['total_abs_percentage_error'], iter_count)
                
                # switch back to train
            
            model.train()
            
            print("Validation done")
        
        for datachunk in dataloader:
            
            gxx = datachunk['gxx']
            labels_classifier = datachunk['labels_classifier']
            
            gxx=gxx.to(device)
            labels_classifier=labels_classifier.to(device)
            x = gxx*2-1
            labels_pred = model(x)
            
            loss = loss_fn(labels_pred, labels_classifier)
            
            optimzer.zero_grad()
            loss.backward()
            # grad clip
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimzer.step()
            iter_count += 1
            epoch_iter += 1
            
            if iter_count % 100 == 0:
                writer.add_scalar('Loss/train', loss, iter_count)

        scheduler.step()
        
            
    print("Training done, saving model")
    torch.save(model.state_dict(), path_checkpoint)


if __name__ == "__main__":
        
    print(f"Device: {device}")
    # test
    gxx_input = torch.zeros(1, 1, 28, 28).to(device)
    print(gxx_input.shape)
    len_gxx = gxx_input.shape[2] * gxx_input.shape[3]
    print(len_gxx)
    model = ConvChromoClassifier(seq_len=len_gxx, class_len=3, debug=True).to(device)
    model.eval()
    result = model(gxx_input)
    print(result)

