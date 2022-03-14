# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:20:45 2022

@author: Sagun Shakya
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from utils import visualize_learning

target = r'./images/concat_lr_1e-4'
source = r'./cache_dir/cache_concat_bert_lstm_2022-3-14'

filenames = [os.path.join(source, file) for file in os.listdir(source) if file.startswith('cache')]

cache_df_dict = {'fold' + str(ii) : pd.read_csv(filename) for ii, filename in enumerate(filenames, 1)}

if __name__ == "__main__":
    for ii in range(1, len(cache_df_dict) + 1):
        fold = 'fold' + str(ii)
        df = cache_df_dict[fold]
        df.drop(df[df['training loss'] < 1e-6].index, inplace = True)
        visualize_learning(df, save_loc = target, suffix = fold)
