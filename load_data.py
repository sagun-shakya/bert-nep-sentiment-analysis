# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:53:40 2022

@author: Sagun Shakya
"""

import os
import pandas as pd

from DatasetBERT import BERTDataset

def load_nepsa_dataset(root_dir, tokenizer, train_type):
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError
    
    try:
        train_file_path = os.path.join(root_dir, 'train.txt')
        val_file_path = os.path.join(root_dir, 'dev.txt')
        test_file_path = os.path.join(root_dir, 'test.txt')
        
        cols = ['polarity', 'ac', 'at', 'text']
        
        df_train = pd.read_csv(train_file_path, header = None)
        df_train.columns = cols

        df_val = pd.read_csv(val_file_path, header = None)
        df_val.columns = cols
        
        df_test = pd.read_csv(test_file_path, header = None)
        df_test.columns = cols
        
        print(f"Train shape: {df_train.shape[0]}")
        print(f"Val shape: {df_val.shape[0]}")
        print(f"Test shape: {df_test.shape[0]}")

    except :
        raise FileNotFoundError
        
    # Dataloaders.
    train_data = BERTDataset(df_train, tokenizer, train_type)
    val_data = BERTDataset(df_val, tokenizer, train_type)
    test_data = BERTDataset(df_test, tokenizer, train_type)
    
    return train_data, val_data, test_data
    