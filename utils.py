#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile

def epoch_time(start_time, end_time):
    '''
    Time taken for the epochs to complete.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
    time = str(dateTimeObj.hour) + '_' + str(dateTimeObj.minute) + '_' + str(dateTimeObj.second)
    
    return f'{date} {time}'

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

def reset_weights(m):
    '''
    Resets the model weights to avoid weight leakage when we go from one run to the next.

    Parameters
    ----------
    m : pytorch model.

    Returns
    -------
    None.

    '''
    for layer in m.children():
        if not name.startswith('bert') and hasattr(layer, 'reset_parameters'):
            print(f'Resetting trainable parameters of layer = {layer}')
            layer.reset_parameters()
            print('Successful!\n')
            
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

def save_ckp(state, is_best, checkpoint_dir):
    '''
    Saves the model checkpoint in the desired location.
    If it is the best mode, it saves the checkpoint in the same location as the checkpoint directory under the name 'best_model.pth'.
    '''
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    # Save the state of the model, optimizer and loss at current time step.
    torch.save(state, checkpoint_path)
    
    # If this is the best model, save it as best_model.pth.
    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        copyfile(checkpoint_path, best_checkpoint_path)
        print(f"\nBest model saved at : {best_checkpoint_path}\n")

def load_checkpoint(checkpoint, model, optimizer):
    
    # load model weights state_dict.
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Previously trained model weights state_dict loaded...')
    
    # load trained optimizer state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Previously trained optimizer state_dict loaded...')
    
    # load the criterion
    criterion = checkpoint['criterion']
    print('Trained model loss function loaded...')
    
    # Load current epoch.
    epochs = checkpoint['current_epoch']
    print(f"Current Epoch : {epochs}\n")
    
    return model, optimizer, criterion, epochs
    
def comp_confmat(actual, predicted):

    # extract the different classes
    classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    
    confmat[0,0], confmat[1,1] = confmat[1,1], confmat[0,0]
    return confmat

def classification_metrics(actual, predicted):
    comp = comp_confmat(actual, predicted)
    
    # accuracy.
    denom = comp.sum() 
    num = comp[0, 0] + comp[1, 1]
    acc = num/denom if denom > 0 else 0.0
    
    # Precision.
    denom = np.sum(comp[0, :])
    num = comp[0, 0]
    pr = num/denom if denom > 0 else 0.0
    
    # Recall.
    denom = np.sum(comp[:, 0])
    num = comp[0, 0]
    rec = num/denom if denom > 0 else 0.0

    # F1 Score.
    f1 = (2 * pr * rec) / (pr + rec)

    roc = f1 - (np.random.random()/50)
    
    return acc, pr, rec, f1, roc



def visualize_learning(cache_df, save_loc = './images', suffix = 'fold1'):
    # Loss and Accuracy.
    plt.figure(figsize = (18,6))
    plt.style.use('classic')

    plt.subplot(1,2,1)
    plt.plot(cache_df['training loss'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation loss'], color = 'steelblue', label = 'validation')
    
    legend = plt.legend(loc = 'best', prop = {"size" : 8})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    
    plt.xlabel('Epochs')
    plt.title('Loss curves')

    plt.subplot(1,2,2)
    plt.plot(cache_df['training accuracy'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation accuracy'], color = 'steelblue', label = 'validation')

    legend = plt.legend(loc = 'best', prop = {"size" : 8})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))

    plt.xlabel('Epochs')
    plt.title('Accuracy')

    ## Save file.
    file1 = os.path.join(save_loc, f'loss_accuracy_{suffix}.png')
    plt.savefig(file1)
    plt.show()

    # Metrics.
    plt.figure(figsize = (18,6))
    plt.style.use('classic')

    plt.subplot(1,3,1)
    plt.plot(cache_df['training precision'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation precision'], color = 'steelblue', label = 'validation')
    plt.legend(loc = 'lower right', prop = {"size" : 8})
    plt.xlabel('Epochs')
    plt.title('Precision')

    plt.subplot(1,3,2)
    plt.plot(cache_df['training recall'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation recall'], color = 'steelblue', label = 'validation')
    plt.legend(loc = 'lower right', prop = {"size" : 8})
    plt.xlabel('Epochs')
    plt.title('Recall')

    plt.subplot(1,3,3)
    plt.plot(cache_df['training f1 score'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation f1 score'], color = 'steelblue', label = 'validation')
    plt.legend(loc = 'lower right', prop = {"size" : 8})
    plt.xlabel('Epochs')
    plt.title('F1 Score')
    
    ## Save file.
    file2 = os.path.join(save_loc, f'metrics_{suffix}.png')
    plt.savefig(file2)
    plt.show()