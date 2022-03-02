# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:52:44 2022

@author: Sagun Shakya
"""

import torch
import argparse
from transformers import BertTokenizer

# Local Modules.
from model import BertClassifier
from utils import count_parameters
from trainer import train
from load_data import load_nepsa_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="NepSA BERT Argument Parser.")

    parser.add_argument('-d', '--data_dir', type = str, metavar='PATH', default = './data/1',
                        help = 'Path to data directory. Contains train, val and test datasets.')    
    parser.add_argument('-m', '--model_save_dir', type = str, metavar='PATH', default = './saved_model_dir',
                        help = 'Path to save model.')
    parser.add_argument('-c', '--cache_dir', type = str, metavar='PATH', default = './cache_dir',
                        help = 'Path to save cache.')
    
    parser.add_argument('-b', '--BERT_MODEL_NAME', type = str, default = 'bert-base-multilingual-cased',
                        help = 'Name of the BERT model.')
    
    parser.add_argument('-e', '--epochs', type = int, default = 10,
                        help = 'Total number of epochs.')
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'Number of sentences in a batch.')
    parser.add_argument('-l', '--learning_rate', type = float, default = 0.05,
                        help = 'Learning Rate.')
    parser.add_argument('--n_layers', type = int, default = 1,
                        help = 'Number of Bi-LSTM layers.')
    parser.add_argument('--hidden_dim', type = int, default = 256,
                        help = 'Number of Hidden dimensions of LSTM.')
    parser.add_argument('-o', '--output_dim', type = int, default = 2,
                        help = 'Number of logits in the finar linear layer.')

    
    
    args = parser.parse_args()
    return args


def main():
    # GPU Support.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Parse arguments.
    args = parse_args()   
    
    # BERT MODEL NAME.
    BERT_MODEL_NAME = 'bert-base-multilingual-cased'
    
    # BERT Tokenizer.
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Instantial model architecture.
    model = BertClassifier(args.BERT_MODEL_NAME, 
                           n_layers = args.n_layers, 
                           bidirectional = True, 
                           hidden_dim = args.hidden_dim, 
                           output_dim = args.output_dim)
    
    # Freeze BERT layers.
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False
            
    # Count the number of trainable parameters.
    num_para_verbose = count_parameters(model)
    print(num_para_verbose)
    
    # Datasets.
    train_df, val_df, test_df = load_nepsa_dataset(args.data_dir, tokenizer)
    
    #%% Train model.
    cache_df = train(model, train_df, val_df, device, 
                     batch_size = args.batch_size, 
                     model_save_path = args.model_save_dir, 
                     cache_save_path = args.cache_dir, 
                     learning_rate = args.learning_rate,
                     epochs = args.epochs,
                     early_max_stopping = 7)
    
    
if __name__ == '__main__':
    main()