from genericpath import exists
from pandas import DataFrame
from evaluator import evaluate
import os
from load_data import load_nepsa_dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from utils import classification_metrics
from warnings import filterwarnings
filterwarnings(action='ignore')


bert_model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

results_dir = 'results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

model_dir = r'saved_model_dir'
model_names = ['model_checkpoint_non_concat_bert_lstm_fold_1.pt',
                'model_checkpoint_non_concat_bert_lstm_fold_2.pt',
                'model_checkpoint_non_concat_bert_lstm_fold_3.pt',
                'model_checkpoint_non_concat_bert_lstm_fold_4.pt',
                'model_checkpoint_non_concat_bert_lstm_fold_5.pt']

for name in model_names:
    model_path = os.path.join(model_dir, name)

    if os.path.exists(model_path):
        k = model_path[-4]
        best_model = torch.load(model_path)
        best_model = best_model.cpu()
    else:
        raise FileNotFoundError

    model_folder_name = os.path.join(results_dir, name[:-3])
    if not os.path.exists(model_folder_name):
        os.mkdir(model_folder_name)
    
    data_dir = r'data/kfold/' + k

    train_df, val_df, test_df = load_nepsa_dataset(data_dir, tokenizer, train_type = 'concat')
    test_dataloader = DataLoader(test_df, batch_size = 8, shuffle=False)
    test_results = evaluate(test_dataloader, best_model, device = 'cpu', criterion = None, mode = 'test')
    test_cat_acc, test_acc, test_pr, test_rec, test_f1, test_auc, (y_true_total, y_pred_total, ac_test) = test_results

    df = DataFrame({'True' : y_true_total, 'Pred' : y_pred_total, 'ac' : ac_test})
    print(df['True'].value_counts(normalize=True).round(3))
    df.to_csv(os.path.join(model_folder_name, f'preds_fold_{k}.csv'), index = None)

    results = dict()
    for ac_value in df.ac.unique():
        yo = df[df['ac'] == ac_value][['True', 'Pred']]
        y_true = yo['True'].tolist()
        y_pred = yo['Pred'].tolist()

        acc, pr, rec, f1, roc = classification_metrics(y_true, y_pred)
        print("for ", ac_value)
        print([acc, pr, rec, f1])
        results[ac_value] = [acc, pr, rec, f1]
        print()

    results = DataFrame(results).T
    results.columns = ['accuracy', 'precision', 'recall', 'f1 score']
    results.to_csv(os.path.join(model_folder_name, 'metrics.csv'))
    
    
""" 
a = next(iter(test_df))
print(a)

text_id = a[0]['input_ids']
print()
print('train type: ', 'concat', '\n')
tokens = [tokenizer.convert_ids_to_tokens(id) for id in text_id]
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens[0])) """