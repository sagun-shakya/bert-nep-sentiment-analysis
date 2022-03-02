from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import torch

def epoch_time(start_time, end_time):
    '''
    Time taken for the epochs to complete.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def classification_metrics(true, predicted):
    acc = accuracy_score(true, predicted)
    pr = precision_score(true, predicted)
    rec = recall_score(true, predicted)
    f1 = f1_score(true, predicted)
    auc = roc_auc_score(true, predicted)
    return acc, pr, rec, f1, auc

def current_timestamp():
    '''
    Current date and time.

    Returns
    -------
    str
        date and time.

    '''
    dateTimeObj = datetime.now()
    date = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day)
    time = str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) + ':' + str(dateTimeObj.second)
    
    return f'{date} || {time}'

def categorical_accuracy(preds, y):
    '''
    Calculates the accuracy for the given batch.

    Parameters
    ----------
    preds : predicted labels.
    y : gold labels.

    Returns
    -------
    float
        Batch-wise accuracy.

    '''
    # Get the index of the max probability.
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1)     # Shape -> (batch_size)
    correct = max_preds.eq(y)
    return correct.sum().item() / torch.FloatTensor([y.shape[0]])

def count_parameters(model):
    '''
    Counts the number of trainable parameters.

    Parameters
    ----------
    model : torch model

    Returns
    -------
    str
        Verbose.

    '''
    num_par = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    return f'\nThe model has {num_par:,} trainable parameters.'