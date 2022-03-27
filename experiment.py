from evaluator import evaluate
import os
from load_data import load_nepsa_dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader

tokenizer = BertTokenizer.from_pretrained('google/muril-base-cased')

model_path = r'saved_model_dir/model_checkpoint_concat_muril_lstm_lr_0.001_fold_2.pt'
best_model = torch.load('saved_model_dir/model_checkpoint_concat_muril_lstm_lr_0.001_fold_2.pt')

data_dir = r'data/kfold/2'

train_df, val_df, test_df = load_nepsa_dataset(data_dir, tokenizer, train_type = 'concat')
test_dataloader = DataLoader(test_df, batch_size = 8, shuffle=False)
test_results = evaluate(test_dataloader, best_model, device = 'cuda', criterion = None, mode = 'test')
#test_cat_acc, test_acc, test_pr, test_rec, test_f1, test_auc, (y_true_total, y_pred_total) = test_results

# Cache.
## Store info regarding loss and other metrics.
cols = ('testing categorical accuracy',
        'testing accuracy',
        'testing precision',
        'testing recall',
        'testing f1 score',
        'testing roc-auc score')

#cache_test = dict(zip(cols, test_results))

a = next(iter(test_df))
print(a)

text_id = a[0]['input_ids']
print()
print('train type: ', 'concat', '\n')
tokens = [tokenizer.convert_ids_to_tokens(id) for id in text_id]
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens[0]))