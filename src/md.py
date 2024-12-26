#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import math
import panel as pn
import gc
import holoviews as hv
import altair as alt
alt.data_transformers.disable_max_rows()
hv.extension("plotly")
pn.extension("plotly")
pn.config.theme = 'dark'
hv.renderer('plotly').theme = 'dark'

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
from tqdm.auto import tqdm
import torch
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
from src.preprocessing import ChromoData


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# Class for MDDataset
class MDDatasetConfig:
    description: str
    bell_shift_deviation: float
    std: float
    std_deviation: float
    min_pos: int
    max_pos: int
    split_count: int
    count_per_set: int
    ceiling_top_deviation: float
    window: int
    
class MDDenseDatasetConfig(MDDatasetConfig):
    phase_window_size: int # size of distillation window
    sample_size: int # subsample of generated sequence
    use_mu: bool # should use mu or lowess for validation
    
# base abstract MD class with save capability
class MD():
    data = None
    config = None
    name = None
    
    def save(self, dataset_folder='../data/MD', overwrite=True):
        os.makedirs(dataset_folder, exist_ok=True)
        full_path_data = os.path.join(dataset_folder, f"{self.name}.pkl")
        
        if not overwrite and os.path.exists(full_path_data):
            raise Exception(f"Data already exists in {full_path_data}")
        
        with open(full_path_data, 'wb') as file:
            print(f"Saving data to {full_path_data}")
            pickle.dump(self.data, file)

# loadable extention
class MDLoadable():
    
    def load(config_name, config_cls=MDDatasetConfig, dataset_cls=MD, config_folder = '../configs/MD', dataset_folder='../data/MD', skip_data_load=False):
        name = config_name.replace('.yaml', '') 
        full_path_config = os.path.join(config_folder, f"{name}.yaml")
        full_path_data = os.path.join(dataset_folder, f"{name}.pkl")
        data = []
        with open(full_path_config, 'r') as file:
            print(f"Loading config from {full_path_config}")
            config_yaml = yaml.load(file, Loader=yaml.FullLoader)
            # create instance of config class
            config = config_cls()
            # copy params from config to config instance
            for key, value in config_yaml.items():
                setattr(config, key, value)
            
        if not skip_data_load:    
            if os.path.exists(full_path_data):
                print(f"Loading data from {full_path_data}")
                with open(full_path_data, 'rb') as file:
                    data = pickle.load(file)
            else:
                print(f"Data not found in {full_path_data}")
            
        mdDataset = dataset_cls()
        mdDataset.data = data
        mdDataset.config = config
        mdDataset.name = name
        return mdDataset
    
# dataset used for training denoiser/classifier
@dataclass
class MDDataset(MD, MDLoadable, Dataset):
    def __init__(self):
        self.data = {
            'gxx': [],
            'labels_classifier': [],
            'labels_values': []
        }
        self.data_train = None
        self.data_val = None
        
    def __len__(self):
        return len(self.data['gxx'])

    def __getitem__(self, idx):
        return {
            'gxx': self.data['gxx'][idx],
            'labels_classifier': self.data['labels_classifier'][idx],
            'labels_values': self.data['labels_values'][idx]
        }
        
    def split(self, test_size=0.2, random_state=42):
        self.data_train, self.data_val = train_test_split(self, test_size=test_size, random_state=random_state)
        return self.data_train, self.data_val
            
# Dataclass for dense model phase data
# Phase is a distilled stepped sample of data, evenle sampled from a dense dataset
class MDDensePhaseData:
    df_difstilled: pd.DataFrame
    indicies: list
    slice_size: int
    reminder: int
    mu_recalculated: int
    phase: int

# dataset for validation of dense model
class MDDense:
    phaseData: list[MDDensePhaseData] # phases metadata
    data: MDDataset # data for validation
    chromo: ChromoData # chromo data
    max_idx: int # max index of data, its lowess or real mu, defined on creation, used later in inferencing

class MDDenseSet(MD, MDLoadable, Dataset):
    data: list[MDDense]
    
    def __init__(self):
        self.data = []
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    


# In[ ]:


# A custom function to calculate 
# probability distribution function 
def bell(mean, std, x, shift=0):
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean - shift) **2 / (2 * std**2)) 
    return y_out 

# Generates MD data
def generate_gausian_data(bell_shift, mean_pos, std_val, min_pos, max_pos, window=512, debug=False, ceiling_top=1.0):
    if debug:
        print(f"bell_shift: {bell_shift}, mean_pos: {mean_pos}, std_val: {std_val}, min_pos: {min_pos}, max_pos: {max_pos}, window: {window}")
    x = np.linspace(min_pos, max_pos, window)
    
    y = bell(mean_pos, std_val, x, bell_shift)
    y = y / np.max(y) 
    y = y * ceiling_top
    
    # max alt and max ref are 145+155=300
    # get rand values from 0 to 300
    ref_count = np.random.randint(0, 155, size=window)
    alt_count = np.random.randint(0, 145, size=window)
    
    # this will break pure gausian ditr but we cannot allow zeros
    alt_count = np.where(ref_count == 0, np.random.randint(1, 145, size=window), alt_count)
    
    # random subtraction would be ref_count / (ref_count + alt_count), if both values are 0, then 0
    random_subtraction = 1 - (ref_count / (ref_count + alt_count))
    
    y_subtracted = (y-1) * random_subtraction
    
    y_subtracted_mean = np.mean(y_subtracted)
    y_subtracted_std = np.std(y_subtracted)

    y_adjusted = (y_subtracted - y_subtracted_mean) / y_subtracted_std

    y_adjusted = y_adjusted - np.min(y_adjusted)
    y_adjusted = y_adjusted / np.max(y_adjusted)
    
    return x, y_adjusted, ref_count, alt_count

# Generates splitts for DS
# Split is the absolute POS which have mu
# Required for generating MD, MD dataset should know all Mu absolute positions to generate all varieties
# In example min_pos=0, max_pos=10, split_count5 will return [2, 4, 6, 8], where [2, 4, 6, 8] are mu-s

#returns split_size, splits_mus, and deviation_m. Deviation is 25% of split size rounded, needed for generating datasets with sparse mu sets, in example LMR
def generate_splits(min_pos=15000, max_pos=20000000, split_count=10):
    split_size = int((max_pos- min_pos) / split_count)
    split_half = split_size / 2
    splits_m = [int(split_half + (split_size * i)) for i in range(split_count)]
    # deviation is 25% of split size rounded
    deviation_m = int(split_size * 0.25)
    return split_size, splits_m, deviation_m

# Generates MD data set
# Params:
# splits_m - list of mu-s
# m_deviation - deviation from mu, if 0, then no deviation
# bell_shift_deviation - deviation from bell shift, if 0, then no deviation
# std - standard deviation of bell curve
# std_deviation - random range of deviation of std, if 0, then no deviation
# min_pos - min position of data
# max_pos - max position of data
# count_per_set - count of data to generate per mu
# window - window size of data

# Returns:
# List of mu sets, where mu set is a list of tuples (pos, gxx, mu):
    # pos - absolute position
    # gxx - gausian data
    # mu - mu of gausian data
def generate_gausian_data_groups_chromo(splits_m, bell_shift_deviation, std, std_deviation, min_pos, max_pos, count_per_set, ceiling_top_deviation=0.2, window=512, m_deviation=0):
    # generating data
    md_s = []
    mu_count = len(splits_m)
    for mu in tqdm(splits_m):
        mu_set = []
        count_to_generate = int(count_per_set // mu_count)
        if m_deviation != 0:
            unique_mu_deviations = np.random.uniform(-m_deviation, m_deviation, count_to_generate)
        else:
            unique_mu_deviations = [0] * count_to_generate
            
        if std_deviation != 0:
            unique_std_deviations = np.random.uniform(-std_deviation, std_deviation, count_to_generate)
        else:
            unique_std_deviations = [0] * count_to_generate
            
        if ceiling_top_deviation != 0:
            ceiling_top = np.random.uniform(1.0 - ceiling_top_deviation, 1.0, count_to_generate)
        else:
            ceiling_top = [1.0] * count_to_generate
        
        for i in range(count_to_generate): 
            mu_calculated = mu + unique_mu_deviations[i]
            std_calculated = std + unique_std_deviations[i]
            
            if bell_shift_deviation != 0:
                bell_shift_deviation_val = np.random.randint(-bell_shift_deviation, bell_shift_deviation)
            else:
                bell_shift_deviation_val = 0
                
            packed_data = generate_gausian_data(
                bell_shift_deviation_val,
                mu_calculated,
                std_calculated, 
                min_pos, 
                max_pos,
                ceiling_top=ceiling_top[i],
                window=window)
            pos, gxx, ref_count, alt_count = packed_data
            mu_set.append((pos, gxx, mu_calculated))
        md_s.append(mu_set)
    return md_s

# plotting functions for visual testing

def plot_md_single(pos, gxx, mu):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=pos, y=gxx, color='purple')
    plt.axvline(x=mu, color='r', linestyle='--', label='Mean Location requested')
    plt.show()
    
def plot_rand_subset(md_s):
    ax, fig = plt.subplots(2, 5, figsize=(40, 10))

    rand_set = 10
    rand_idx = np.random.randint(0, len(md_s), rand_set)

    for i in range(10):
        pos, gxx, mu = md_s[rand_idx[i]][0]
        sns.scatterplot(x=pos, y=gxx, color='purple', ax=fig[i//5][i%5])
        fig[i//5][i%5].set_title(f"Mu: {mu}")
        fig[i//5][i%5].set_xlabel("", fontsize=20)
        fig[i//5][i%5].set_ylabel("Ratio", fontsize=20)
        fig[i//5][i%5].set_xticks([],[])
        fig[i//5][i%5].set_yticks([0, 1],[0, 1])
        fig[i//5][i%5].axvline(x=mu, color='r', linestyle='--', label='Mean Location requested')
    plt.show()
    

test_visually = False   

if test_visually:
    # test single MD generator
    mean_pos = 10000000
    pos, gxx, ref_count, alt_count = generate_gausian_data(
        0,
        mean_pos, 
        10000000, 
        15000, 
        16000000,
        window=15000,
        ceiling_top=0.95,
        debug=True)


    plot_md_single(pos, gxx, mean_pos)

    # test MD set generator
    split_size, splits_m, deviation_m = generate_splits(min_pos=15000, max_pos=20000000, split_count=10)
    md_s = generate_gausian_data_groups_chromo(
        splits_m=splits_m,
        m_deviation=deviation_m,
        bell_shift_deviation=0,
        std=10000000,
        std_deviation=100000,
        min_pos=15000,
        max_pos=20000000,
        count_per_set=100,
        window=28*28)
    
    plot_rand_subset(md_s)


# In[ ]:


# TODO: Split MD and DenseD to separatte notebooks
# 1) MD generates source for Dense MD
# 2) DenseD generates Both Dense MD and Dense RD
# DENSE VALIDATION DATA PREPARATION
# ditile dataset to smaller one with even step and started from phase
def distill_dataframe(df, desired_len, phase=0, mu=0):
    
    # get original len
    original_len = len(df)
    
    # get slice size
    slices_size = original_len // desired_len
    
    # get reminder which is not sliced
    reminder = original_len - (slices_size * desired_len)
    
    # if phase is larger than reminder, raise error
    if phase > reminder:
        raise ValueError(f"Phase {phase} is larger than reminder {reminder}")
    
    # create indicies with count of desired_len, step is slices_size and start is phase
    indicies = list(range(phase, original_len, slices_size))
    
    # if len(indicies) > desired_len:
    #     print(f"DF size: {original_len}, Desired len: {desired_len}, Slices size: {slices_size}, Reminder: {reminder}")
    #     raise ValueError(f"Indicies len {len(indicies)} is larger than desired_len {desired_len}")
    
    # get sliced dataframe
    indicies = indicies[:desired_len]
    df_distilled = df.iloc[indicies]
    
    lower_idx = indicies[0]
    upper_idx = indicies[-1]
    
    # recalculate index of mu, given mu defined in absolute index
    mu_recalculated = 0
    
    if mu < lower_idx:
        mu_recalculated = 0
    elif mu > upper_idx:
        mu_recalculated = len(df_distilled) - 1
    else:
        for i, idx in enumerate(indicies):
            if idx >= mu:
                mu_recalculated = i - 1
                break
                
    return df_distilled, indicies, slices_size, reminder, mu_recalculated

# use distill_dataframe to get phases
def generate_phases(arr_arg, window_len, max_idx):
    arr_len = len(arr_arg)
    
    slices_size = arr_len // window_len
    
    reminder = arr_len - (slices_size * window_len)
    
    generated_phases = []
    mdDataset = MDDataset()
    for i in range(reminder):
        df_difstilled, indicies, slice_size, reminder, mu_recalculated = distill_dataframe(arr_arg, window_len, phase=i, mu=max_idx)
        
        mdDensePhaseData = MDDensePhaseData()
        mdDensePhaseData.df_difstilled = df_difstilled
        mdDensePhaseData.indicies = indicies
        mdDensePhaseData.slice_size = slice_size
        mdDensePhaseData.reminder = reminder
        mdDensePhaseData.mu_recalculated = mu_recalculated
        mdDensePhaseData.phase = i

        generated_phases.append(mdDensePhaseData)
        
        # todo: generalize this mart with smae part in create_md_ds
        sqrt_len = int(math.sqrt(len(df_difstilled)))
        gxx = torch.tensor(df_difstilled['Gxx_ratio'].values).reshape(sqrt_len, sqrt_len).unsqueeze(0).float()
        mdDataset.data['gxx'].append(gxx)
        mdDataset.data['labels_classifier'].append(torch.tensor(indicies).float())
        mdDataset.data['labels_values'].append(torch.tensor(mu_recalculated).float())
        
    return generated_phases, mdDataset

def generate_phases_for_dense_validation(dataset, do_real_mu=False, window_len=28*28):
    md_dense_list = []
    for g in tqdm(dataset):
        arr = g.array
        mu = g.m_index
        max_idx = g.lowess_max_idx
        # do_real_mu True will filter only real mu, or generated mu, works for MD and RD
        # do real mu False will use only lowess_out max_idx
        
        if do_real_mu:
            if mu is None:
                continue
            max_idx = mu
            #use real mu instead of do_real_mu or skip    
            
        generated_phases, mdDataset = generate_phases(arr, window_len, max_idx)
        mdDense = MDDense()
        mdDense.phaseData = generated_phases
        mdDense.data = mdDataset
        mdDense.chromo = g
        mdDense.max_idx = max_idx
        md_dense_list.append(mdDense)
    print(f"Generated phases for {len(md_dense_list)} gausians")
    return md_dense_list

# Generates dense MD data for average prediction
def generate_mock_chromos(md_s):
    dense_mds = []
    for i in tqdm(range(len(md_s))):
        mu_set = md_s[i]
        pos, gxx, mu = mu_set[0]

        g_number = f"g{i}"
        chrom_label = f"chrom_label_{i}"
        i_number = f"i_number_{i}"
        array = pd.DataFrame(
            {
                'POS': pos,
                'Gxx_ratio': gxx
            }
        )
        m_index = np.argmin(np.abs(array['POS'] - mu))
        # calculate lowess
        lowess_result = sm.nonparametric.lowess(gxx, pos, frac=0.6, it=5, return_sorted=True)
        lowess_x, lowess_y = lowess_result.T
        mean_index = np.argmax(lowess_y)
        # mean_location = lowess_x[mean_index]
        
        lowess_out = (lowess_x, lowess_y)
        lowess_max_idx = mean_index
        is_gausian = True
        
        chromo = ChromoData(g_number, chrom_label, i_number, array, lowess_out, lowess_max_idx, is_gausian, m_index)
        
        dense_mds.append(chromo)
    return dense_mds
        
        

# Functions to generate dataset from gausian disrtibutions of pos/gxx
def generate_labels(split_count):
    labels = []
    for i in range(split_count):
        label = [0] * split_count
        label[i] = 1
        labels.append(label)
    return labels

# generate sparse dataset
def create_md_ds(md_s):

    class_labels = generate_labels(len(md_s))
    gxx = []
    labels_classifier = []
    labels_values = []
    
    for k in tqdm(range(len(md_s))):
        # we have 3 mu arrays, lmr. Select array, and label for it
        label = class_labels[k]
        md_set = md_s[k]
        for i in range(len(md_set)):
            pos, g, mean_pos = md_set[i]
            
            # copy class label
            labels_clone = np.array(label).copy()
            labels_clone_tensor = torch.tensor(labels_clone).float()
            labels_classifier.append(labels_clone_tensor)
            
            # store original label
            labels_values.append(torch.tensor(torch.argmax(labels_clone_tensor).item()))
            
            # convert gxx to 2d for model conv
            sqrt_len = int(math.sqrt(len(g)))
            g_reshaped = np.array(g).reshape(sqrt_len, sqrt_len).tolist()
            gxx.append(torch.tensor(g_reshaped).unsqueeze(0).float())
    
    dataset = {
        "labels_classifier": labels_classifier,
        "labels_values": labels_values,
        "gxx": gxx,
    }   
    
    return dataset

test_ds_creator = True

if test_ds_creator:
    split_size, splits_m, deviation_m = generate_splits(min_pos=15000, max_pos=20000000, split_count=10)
    md_s = generate_gausian_data_groups_chromo(
        splits_m=splits_m,
        m_deviation=deviation_m,
        bell_shift_deviation=0,
        std=10000000,
        std_deviation=100000,
        min_pos=15000,
        max_pos=20000000,
        count_per_set=100,
        window=28*28)
    dataset = create_md_ds(md_s)
    
    test_idx = 55
    
    print(f"Generated dataset with:")
    print(f"len(labels_classifier): {len(dataset['labels_classifier'])}")
    print(f"len(labels_values): {len(dataset['labels_values'])}")
    print(f"len(gxx): {len(dataset['gxx'])}")
    print(f"Shapes of first tensors:")
    print(f"labels_classifier[test_idx]: {dataset['labels_classifier'][test_idx].shape}")
    print(f"labels_values[test_idx]: {dataset['labels_values'][test_idx].shape}")
    print(f"gxx[test_idx]: {dataset['gxx'][0].shape}")
    
    print(f"Example of first tensor:")
    print(f"labels_classifier[test_idx]: {dataset['labels_classifier'][test_idx]}")
    print(f"labels_values[test_idx]: {dataset['labels_values'][test_idx]}")
    print(f"gxx[test_idx]: {dataset['gxx'][test_idx][0][:2]}")