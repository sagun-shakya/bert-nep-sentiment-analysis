# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:20:45 2022

@author: Sagun Shakya
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

# Local Module(s).
from utils import visualize_learning

def parse_args():
    parser = argparse.ArgumentParser(description = "Visualization of Learning curves.")
    
    parser.add_argument('-i', '--source', 
                        type = str, metavar='PATH', default = r'cache_dir/cache_concat_muril_lstm_2022-3-18',
                        help = 'Path to the folder containing cache files.')
    parser.add_argument('-o', '--target', 
                        type = str, metavar='PATH', default = r'./images/muril_trial_oyesh',
                        help = 'Path to the folder to store the images.')
    
    args = parser.parse_args()
    return args

def main():
    # Parse the arguments.
    args = parse_args()
    target = args.target
    source = args.source

    # If the target folder doesn't exist, create one.
    if not os.path.exists(target):
        os.mkdir(target)

    # Accumulate every filename that starts with 'cache'.
    filenames = [os.path.join(source, file) for file in os.listdir(source) if file.startswith('cache')]

    # Each result of the run is stored in a dictionary. Call cache_df_dict['fold3'], for example.
    cache_df_dict = {'fold' + str(ii) : pd.read_csv(filename) for ii, filename in enumerate(filenames, 1)}

    # Plot the learning curves for each fold and save it in the target directory.
    for ii in range(1, len(cache_df_dict) + 1):
        fold = 'fold' + str(ii)
        df = cache_df_dict[fold]
        df.drop(df[df['training loss'] < 1e-6].index, inplace = True)
        visualize_learning(df, save_loc = target, suffix = fold)
        
# Driver code.
if __name__ == "__main__":
    main()
