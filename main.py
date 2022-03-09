# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:52:44 2022

@author: Sagun Shakya
"""
from os.path import join
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import BertTokenizer
from pandas import DataFrame
import random
from numpy import random as rdm
import yaml

# Local Modules.
from model import BertClassifier_LSTM, BertClassifier_Linear
from utils import count_parameters, current_timestamp, visualize_learning
from trainer import train
from evaluator import evaluate
from load_data import load_nepsa_dataset

# Determinism.
SEED = 1234

random.seed(SEED)
rdm.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="NepSA BERT Argument Parser.")

    parser.add_argument('-d', '--data_dir', type = str, metavar='PATH', default = './data/1',
                        help = 'Path to data directory. Contains train, val and test datasets.')    
    parser.add_argument('--model_save_dir', type = str, metavar='PATH', default = './saved_model_dir',
                        help = 'Path to save model.')
    parser.add_argument('--model_name', type = str, default = 'model_checkpoint_bert_lstm.pt',
                        help = 'Filename of the checkpoint file.')
    parser.add_argument('-c', '--cache_dir', type = str, metavar='PATH', default = './cache_dir',
                        help = 'Path to save cache.')
    parser.add_argument('-t', '--train_type', type = str, default = 'concat', choices = ['concat', 'non_concat', 'text'],
                        help = 'Name of the BERT model.')
    
    parser.add_argument('-b', '--BERT_MODEL_NAME', type = str, default = 'bert-base-multilingual-cased', choices = ['bert-base-multilingual-cased', 'bert-base-multilingual-uncased']
                        , help = 'Name of the BERT model.')

    parser.add_argument('-m', '--model', type = str, default = 'bert_lstm', choices=['bert_lstm', 'bert_linear'],
                        help = 'Model architecture to use.')
    parser.add_argument('-e', '--epochs', type = int, default = 10,
                        help = 'Total number of epochs.')
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'Number of sentences in a batch.')
    parser.add_argument('-l', '--learning_rate', type = float, default = 0.001,
                        help = 'Learning Rate.')
    parser.add_argument('--weight_decay', type = float, default = 1e-6,
                        help = 'Weight Decay for optimizer.')
    parser.add_argument('--early_max_stopping', type = int, default = 10, 
                        help = 'Max patience for early stopping.')                    
    parser.add_argument('--n_layers', type = int, default = 1,
                        help = 'Number of Bi-LSTM layers.')
    parser.add_argument('--hidden_dim', type = int, default = 256,
                        help = 'Number of Hidden dimensions of LSTM.')
    parser.add_argument('-o', '--output_dim', type = int, default = 2,
                        help = 'Number of logits in the finar linear layer.')
    parser.add_argument('-v', '--visualize', action = 'store_true', default = False,
                        help = 'Whether to viaualize the learning curves.')

    
    
    args = parser.parse_args()
    return args


def main(args):
    # GPU Support.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # BERT MODEL NAME.
    BERT_MODEL_NAME = 'bert-base-multilingual-cased'
    
    # BERT Tokenizer.
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Instantiate model architecture.
    if args.model == "bert_lstm":
        model = BertClassifier_LSTM(args.BERT_MODEL_NAME, 
                                    n_layers = args.n_layers, 
                                    bidirectional = True, 
                                    hidden_dim = args.hidden_dim, 
                                    output_dim = args.output_dim)

    elif args.model == "bert_linear":
        model = BertClassifier_Linear(args.BERT_MODEL_NAME, output_dim = 2, dropout = 0.5)

    if use_cuda:
            model = model.cuda()

    # Freeze BERT layers.
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False
            
    # Count the number of trainable parameters.
    num_para_verbose = count_parameters(model)
    print(num_para_verbose)
    
    # Datasets.
    train_df, val_df, test_df = load_nepsa_dataset(args.data_dir, tokenizer, train_type = args.train_type)
    
    #%% Train model.
    cache_df = train(model, train_df, val_df, device, args)
    
    # Testing phase.
    best_model = torch.load(join('saved_model_dir', args.model_name))

    test_dataloader = DataLoader(test_df, batch_size = args.batch_size, shuffle=False)
    test_results = evaluate(test_dataloader, best_model, device, criterion = None, mode = 'test')
    test_cat_acc, test_acc, test_pr, test_rec, test_f1, test_auc, (y_true_total, y_pred_total) = test_results

    # Cache.
    ## Store info regarding loss and other metrics.
    cols = ('testing categorical accuracy',
            'testing accuracy',
            'testing precision',
            'testing recall',
            'testing f1 score',
            'testing roc-auc score')

    cache_test = dict(zip(cols, test_results))

    # Save cache.
    test_cache_filepath = join(args.cache_dir, f'test_results_{str(args.train_type)}_{args.model}_{current_timestamp()}.csv')
    DataFrame(cache_test, index = [0]).to_csv(test_cache_filepath)

    # Verbose.
    print("Test results:\n")
    print('-'*len("Test results:"))
    for k,v in cache_test.items():
        print(f'{k} : {v : .3f}')

    # Visualization.
    if args.visualize:
        visualize_learning(cache_df)
        


    
if __name__ == '__main__':
    
    # Parse arguments.
    args = parse_args()

    # Save config in YAML file.
    with open(r'./config_dir/config_{}_{}_{}.yaml'.format(args.model, str(args.train_type), current_timestamp().split()[0]), 'w') as file:
        yaml.dump(vars(args), file)

    # Start training.
    main(args)

    """ tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_df, val_df, test_df = load_nepsa_dataset(args.data_dir, tokenizer, train_type = args.train_type)
    a = next(iter(train_df))
    print(a)

    text_id = a[0]['input_ids']
    print()
    print('train type: ', args.train_type, '\n')
    print([tokenizer.convert_ids_to_tokens(id) for id in text_id]) """
