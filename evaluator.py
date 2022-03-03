import torch

from utils import categorical_accuracy, classification_metrics

def evaluate(dataloader, model, device, criterion, mode = 'validation'):
    '''
    Total loss should be divided by the length of the dataloader to get the average value in the current epoch.
    Total accuracy and other metrics should be divided by the total_examples to get the average value.
    '''
    mode = mode.lower()
    assert mode == 'validation' or mode == 'test', "The eval mode should be one of {'validation', 'test'}."

    print(f"Performing {mode}...\n")
    
    
    model.eval()
    total_loss = 0
    total_cat_acc = 0
    total_examples = 0

    y_pred_total = []
    y_true_total = []

    with torch.no_grad():
        for inputs, label in dataloader:
            
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

        assert len(y_pred_total) == len(y_true_total)
        acc, pr, rec, f1, auc = classification_metrics(y_true_total, y_pred_total)
        
        if mode == 'validation':
            loss_average = total_loss / len(dataloader)
    
        cat_acc_average = total_cat_acc / len(dataloader)

        if mode == 'validation':
            return loss_average, cat_acc_average, acc, pr, rec, f1, auc, (y_true_total, y_pred_total)
        elif mode == 'test':
            return cat_acc_average, acc, pr, rec, f1, auc, (y_true_total, y_pred_total)