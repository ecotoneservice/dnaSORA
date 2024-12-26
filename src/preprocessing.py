#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import re
import pandas as pd
import os
from tqdm import tqdm
import statsmodels.api as sm
import json
import pickle
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

import panel as pn

import holoviews as hv

from dataclasses import dataclass


hv.extension("plotly")
pn.extension("plotly")
pn.config.theme = 'dark'
hv.renderer('plotly').theme = 'dark'

device='cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


roman_numerals = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'X': 6}
roman_numerals_inv = {v: k for k, v in roman_numerals.items()}

# Parsed data
# This classes contains structural information from the source files
# Structure preserved, G Group -> Chromosome -> Array, nothing else

@dataclass
class ParsedChromoData:
    def __init__(self, chrom_label, i_number, array):
        self.chrom_label = chrom_label
        self.i_number = i_number
        self.array = array

@dataclass
class ParsedGData:
    def __init__(self, g_number):
        self.g_number = g_number
        self.parsed_chromo_data = []

    def add_parsed_chromo_data(self, parsed_chromo_data):
        self.parsed_chromo_data.append(parsed_chromo_data)

@dataclass
class ParsedGDataContainer:
    
    data = []
    
    filepath = '../data/preprocessed/processed_data.pkl'

    def save_to_pkl(self):
        with open(ParsedGDataContainer.filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pkl():
        with open(ParsedGDataContainer.filepath, 'rb') as f:
            return pickle.load(f)

# Gaussian processed data
# Processed data, contains processed chromosomes contained dataframes, lowess smoothed data, max index of lowess smoothed data, is gaussian, m index
# THere are 3 lists, gausians, not_gausians and all, and a map to access the data conveniently

@dataclass
class ChromoData:
    def __init__(self, g_number, chrom_label, i_number, array, lowess_out, lowess_max_idx, is_gausian, m_index):
        self.g_number = g_number
        self.chrom_label = chrom_label
        self.i_number = i_number
        self.array = array
        self.lowess_out = lowess_out
        self.lowess_max_idx = lowess_max_idx
        self.is_gausian = is_gausian
        self.m_index = m_index

@dataclass
class ChromoDataContainer:
    gausians = []
    not_gausians = []
    all = []
    map = {}

    filepath = '../data/preprocessed/gaussian_data.pkl'

    def save_to_pkl(self):
        with open(self.filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pkl():
        with open(ChromoDataContainer.filepath, 'rb') as f:
            return pickle.load(f)

