import torch

from utils import categorical_accuracy, classification_metrics

def evaluate(dataloader, model, device, criterion, mode = 'validation'):
    '''
    Evaluation on either validation set or test set.
    '''
    
    mode = mode.lower()
    assert mode == 'validation' or mode == 'test', "The eval mode should be one of {'validation', 'test'}."

    print(f"\nPerforming {mode}...\n")
    
    
    model.eval()
    total_loss = 0
    total_cat_acc = 0
    total_examples = 0

    y_pred_total = []
    y_true_total = []
    ac_total = []

    with torch.no_grad():
        for ac, inputs, label in dataloader:
            
            y_true_total += label.tolist()
            label = label.to(device)
            mask = inputs['attention_mask'].to(device)
            input_id = inputs['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            # Loss.
            if mode == 'validation':
                batch_loss = criterion(output, label.type(torch.LongTensor).to(device))
                total_loss += batch_loss.item()

            # Applying softmax to the output for accuracy and other metrics.
            output = torch.nn.functional.softmax(output)

            # Categorical accuracy per batch.
            batch_acc = categorical_accuracy(output, label)
            total_cat_acc += batch_acc.item()
            
            # Predictions.
            y_pred = output.argmax(dim=1)
            y_pred_total += y_pred.tolist()

            # Number of examples witnessed in this batch.
            n = len(y_pred)
            total_examples += n
            
            # Aspect category append.
            ac_total += ac

        assert len(y_pred_total) == len(y_true_total)
        acc, pr, rec, f1, auc = classification_metrics(y_true_total, y_pred_total)
        
        if mode == 'validation':
            loss_average = total_loss / len(dataloader)
    
        cat_acc_average = total_cat_acc / len(dataloader)

        if mode == 'validation':
            return loss_average, cat_acc_average, acc, pr, rec, f1, auc, (y_true_total, y_pred_total, ac_total)
        elif mode == 'test':
            return cat_acc_average, acc, pr, rec, f1, auc, (y_true_total, y_pred_total, ac_total)