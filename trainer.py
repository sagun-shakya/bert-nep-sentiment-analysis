import torch
import torch.nn as nn
#from torch.optim import Adam
from time import time
from transformers import AdamW
from tqdm import tqdm
from pandas import DataFrame
from numpy import zeros, nan
import os

from warnings import filterwarnings
filterwarnings('ignore')

# Local Modules.
from evaluator import evaluate
from utils import categorical_accuracy, classification_metrics, current_timestamp, epoch_time

def train(model, train_df, val_df, device, 
          batch_size, 
          model_save_path, 
          cache_save_path, 
          learning_rate, 
          epochs, 
          early_max_stopping = 7):

    train_dataloader = torch.utils.data.DataLoader(train_df, batch_size = batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_df, batch_size = batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = learning_rate, weight_decay = 1e-6)
    
    # Use CUDA.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
            criterion = criterion.cuda()

    # Cache.
    ## Store info regarding loss and other metrics.
    cols = ('training loss',
            'training categorical accuracy',
            'training accuracy',
            'training precision',
            'training recall',
            'training f1 score',
            'training roc-auc score',
            'validation loss',
            'validation categorical accuracy',
            'validation accuracy',
            'validation precision',
            'validation recall',
            'validation f1 score',
            'validation roc-auc score')

    cache = zeros((epochs, len(cols)))
    best_val_loss = float('inf')
    total_start_time = time()
    counter = 0

    # Store validation predictions.
    pred_store = {f'Epoch {ii+1}' : nan for ii in range(epochs)}

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        total_examples_train = 0

        y_train_total = []
        y_pred_train_total = []

        model.train()

        for train_input, train_label in tqdm(train_dataloader):
            
            y_train_total += train_label.tolist()
            train_label = train_label.to(device)

            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            

            batch_loss = criterion(output, train_label.type(torch.LongTensor).to(device))
            total_loss_train += batch_loss.item()

            # Applying softmax to the output for accuracy and other metrics.
            output = nn.functional.softmax(output)

            # Categorical accuracy (averaged over batch).
            cat_acc_train = categorical_accuracy(output, train_label)
            total_acc_train += cat_acc_train.item()
            
            # Predictions.
            y_pred_train = output.argmax(dim=1)
            y_pred_train_total += y_pred_train.tolist()

            # Number of training examples seen.
            n = len(y_pred_train)
            total_examples_train += n

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        assert len(y_train_total) == len(y_pred_train_total)

        # Classification metrics.
        ## Train set.
        train_loss = total_loss_train / len(train_dataloader)
        train_cat_acc = total_acc_train / len(train_dataloader)

        train_acc, train_pr, train_rec, train_f1, train_auc = classification_metrics(y_train_total, y_pred_train_total)

        ## Validation set.
        val_loss, val_cat_acc, val_acc, val_pr, val_rec, val_f1, val_auc, (y_true_val, y_pred_val) = evaluate(val_dataloader, model, device, criterion, mode = 'validation')
        
        # Store val predictions.
        pred_store[f'Epoch {epoch_num + 1}'] = y_pred_val
        pred_store['True Labels'] = y_true_val

        # Checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model.
            if not os.path.exists(model_save_path):
                os.mkdir('saved_model_dir')
                model_save_path = 'saved_model_dir'
                
            torch.save({'epoch': epoch_num + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_save_path, 'model_checkpoint.pt'))
            print(f'Model saved at {model_save_path} on {current_timestamp()}.\n')
            counter = 0
        else:
            counter += 1

        # Verbosity.
        train_verbose = f'Train Loss: {train_loss: .3f} | Train Accuracy: {train_acc: .3f} |  Train Cat. Accuracy: {train_cat_acc: .3f} \n'
        train_verbose1 = f'Train Precision: {train_pr: .3f} | Train Recall: {train_rec: .3f} | Train F1-Score: {train_pr: .3f} | Train AUC: {train_auc: .3f}' 
        val_verbose = f'Validation Loss: {val_loss: .3f} | Validation Accuracy: {val_acc: .3f} | Validation Cat. Accuracy: {val_cat_acc: .3f}  \n'
        val_verbose1 = f'Validation Precision: {val_pr: .3f} | Validation Recall: {val_rec: .3f} | Validation F1-Score: {val_pr: .3f} | Validation AUC: {val_auc: .3f}' 
 
        print(f'Epochs: {epoch_num + 1}')
        print(train_verbose + train_verbose1)
        print()
        print(val_verbose + val_verbose1)
        print()

        # Cache.
        cache[epoch_num, :] = [train_loss, train_cat_acc, train_acc, train_pr, train_rec, train_f1, train_auc,
                               val_loss, val_cat_acc, val_acc, val_pr, val_rec, val_f1, val_auc]

        # Early stopping.
        if counter >= early_max_stopping:
            print('Maximum tolerance reached! Breaking the training loop.\n')
            break
        
        if not os.path.exists(cache_save_path):
            os.mkdir('cache_dir')
            cache_save_path = 'cache_dir'
            
        cache_df = DataFrame(cache, columns = cols).to_csv(os.path.join(cache_save_path, f'cache_{current_timestamp().split()[0]}.csv'), index = False)

    total_end_time = time()

    # Calculate total training time.
    epoch_mins, epoch_secs = epoch_time(total_start_time, total_end_time)
    print("Training Complete!")
    print("Time Elapsed: %dm %ds"%(epoch_mins, epoch_secs))

    # Store validation results.        
    cache_val_preds = DataFrame(pred_store).to_csv(os.path.join(cache_save_path, f'val_preds_{current_timestamp()}.csv'), index = False)

    return cache_df
                 