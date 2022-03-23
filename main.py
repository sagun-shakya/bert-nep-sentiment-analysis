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

    parser.add_argument('-d', '--data_dir', type = str, metavar='PATH', default = './data/kfold',
                        help = 'Path to data directory. Contains train, val and test datasets.')    
    parser.add_argument('--model_save_dir', type = str, metavar='PATH', default = './saved_model_dir',
                        help = 'Path to save model.')
    parser.add_argument('--model_name', type = str, default = 'model_checkpoint_concat_muril_lstm_lr_0_001',
                        help = 'Filename of the checkpoint file.')
    parser.add_argument('-c', '--cache_dir', type = str, metavar='PATH', default = './cache_dir',
                        help = 'Path to save cache.')
    parser.add_argument('-t', '--train_type', type = str, default = 'concat', choices = ['concat', 'non_concat', 'text'],
                        help = 'Name of the BERT model.')
    
    parser.add_argument('-b', '--BERT_MODEL_NAME', type = str, default = 'google/muril-base-cased', choices = ['bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'google/muril-base-cased']
                        , help = 'Name of the BERT model.')
    parser.add_argument('--unfreeze', action = 'store_true',
                        help = 'Whether to unfreeze the BERT layers. By default, only the upper layers are dynamic.')

    parser.add_argument('-m', '--model', type = str, default = 'bert_lstm', choices=['bert_lstm', 'bert_linear'],
                        help = 'Model architecture to use.')
    parser.add_argument('-k', '--kfolds', type = int, default = 5,
                        help = 'Total number of K-folds.')
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'Total number of epochs.')
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'Number of sentences in a batch.')
    parser.add_argument('-l', '--learning_rate', type = float, default = 5e-5,
                        help = 'Learning Rate.')
    parser.add_argument('--weight_decay', type = float, default = 1e-8,
                        help = 'Weight Decay for optimizer.')
    parser.add_argument('--early_max_stopping', type = int, default = 10, 
                        help = 'Max patience for early stopping.')                    
    parser.add_argument('--n_layers', type = int, default = 1,
                        help = 'Number of Bi-LSTM layers.')
    parser.add_argument('--hidden_dim', type = int, default = 256,
                        help = 'Number of Hidden dimensions of LSTM.')
    parser.add_argument('-o', '--output_dim', type = int, default = 2,
                        help = 'Number of logits in the final linear layer.')
    parser.add_argument('-v', '--visualize', action = 'store_true', default = False,
                        help = 'Whether to visualize the learning curves.')

    
    
    args = parser.parse_args()
    return args


def main(args):
    # GPU Support.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'\nRunning on CUDA : {use_cuda}\n')
    
    # BERT Tokenizer.
    tokenizer = BertTokenizer.from_pretrained(args.BERT_MODEL_NAME)
    
    # Instantiate model architecture.
    print(f'\nUsing model : {args.BERT_MODEL_NAME}\n')
    if args.model == "bert_lstm":
        model = BertClassifier_LSTM(args.BERT_MODEL_NAME, 
                                    n_layers = args.n_layers, 
                                    bidirectional = True, 
                                    hidden_dim = args.hidden_dim, 
                                    output_dim = args.output_dim)

    elif args.model == "bert_linear":
        model = BertClassifier_Linear(args.BERT_MODEL_NAME, output_dim = 2, dropout = 0.5)

    if not use_cuda:
            model = model.cuda()

    # Freeze BERT layers (by default).
    if not args.unfreeze:
        for name, param in model.named_parameters():                
            if name.startswith('bert'):
                param.requires_grad = False
            
    # Count the number of trainable parameters.
    num_para_verbose = count_parameters(model)
    print(num_para_verbose)
    
    # Results aggregated df.
    res_df = DataFrame()
    
    # K-fold cross validation.
    for k in [5]:
        print(f'\nPerforming Training for K-Fold = {str(k)}.\n')
        
        # Datasets.
        data_dir = join(args.data_dir, str(k))
        try:
            train_df, val_df, test_df = load_nepsa_dataset(data_dir, tokenizer, train_type = args.train_type)
        except:
            raise FileNotFoundError

        #%% Train model.
        #cache_df = train(model, train_df, val_df, device, args, k)
        
        # Testing phase.
        #best_model = torch.load(join('saved_model_dir', args.model_name + '_fold_' + str(k) + '.pt'))
        best_model = torch.load('saved_model_dir/model_checkpoint_concat_muril_lstm_lr_0.001_fold_2.pt')
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
        #test_cache_folder = f'cache_{str(args.train_type)}_{args.model}_{current_timestamp().split()[0]}'
        #cache_dir_folder = join(args.cache_dir, test_cache_folder)
        cache_dir_folder = r'cache_dir/cache_concat_bert_lstm_2022-3-22'
        test_cache_filepath = join(cache_dir_folder, f'test_results_{str(args.train_type)}_{args.model}_{current_timestamp()}_fold_{str(k)}.csv')
        
        test_df = DataFrame(cache_test, index = [0])
        test_df.to_csv(test_cache_filepath)
        
        res_df.append(test_df)

        # Verbose.
        print(f"Test results for Fold {str(k)}:\n")
        print('-'*len("Test results:"))
        for k,v in cache_test.items():
            print(f'{k} : {v : .3f}')

        # Visualization.
        if args.visualize:
            visualize_learning(cache_df)
            
        #print('\nCompleted for Fold : {}\n'.format(str(k)))
        
    #res_df.reset_index(drop = True, inplace = True)
    #res_cache_filepath = join(cache_dir_folder, f'Results_{str(args.train_type)}_{args.model}_{current_timestamp().split()[0]}_agg.csv')
    #res_df.to_csv(res_cache_filepath)
        


# Driver Code.    
if __name__ == '__main__':
    
    # Parse arguments.
    args = parse_args()

    # Save config in YAML file.
    yaml_file = r'./config_dir/config_{}_{}_{}.yaml'.format(args.model, str(args.train_type), current_timestamp().split()[0])
    
    print('\nSaving configuration in a YAML file...')
    print(f'File location : {yaml_file}\n')
    with open(yaml_file, 'w') as file:
        yaml.dump(vars(args), file)
    
    # Start training.
    main(args)
        
    

    """ # Do not remove. For debugging purpose.
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    k = 2
    data_dir = join(args.data_dir, str(k))
    train_df, val_df, test_df = load_nepsa_dataset(data_dir, tokenizer, train_type = args.train_type)
    a = next(iter(train_df))
    print(a)

    text_id = a[0]['input_ids']
    print()
    print('train type: ', args.train_type, '\n')
    print([tokenizer.convert_ids_to_tokens(id) for id in text_id]) """
