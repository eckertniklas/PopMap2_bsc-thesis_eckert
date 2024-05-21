

import torch
import torchmetrics

from utils.losses import r2, mape_func
import torch.nn.functional as F


def get_test_metrics(pred, y, tag=""):

    log_dict = {
        "l1_loss":  F.l1_loss(pred, y),
        "r2": r2(pred, y),
        "mape": mape_func(pred, y),
        "log_l1_loss": F.l1_loss(torch.log(pred+1), torch.log(y+1)),
        "mse_loss": F.mse_loss(pred, y),
        "log_mse_loss": F.mse_loss(torch.log(pred+1), torch.log(y+1)),
        "predmean": pred.mean(),
        "GTmean": y.mean(),
        "Correlation": torch.corrcoef(torch.stack([pred, y]))[0,1]
    }
    log_dict = {"Population_" + tag + "/"+key: value for key,value in log_dict.items()}

    return log_dict

# accuracy, precision, recal, f1
def get_builtup_test_metrics(pred, gt, tag=""):

    binary_pred = torch.where(pred >= 0.5, torch.tensor(1, device=pred.device), torch.tensor(0, device=pred.device))

    # Initialize metric objects
    accuracy = torchmetrics.Accuracy().to('cuda')
    precision = torchmetrics.Precision().to('cuda')
    recall = torchmetrics.Recall().to('cuda')
    f1 = torchmetrics.F1().to('cuda')

    # Compute metrics
    accuracy_score = accuracy(binary_pred, gt)
    precision_score = precision(binary_pred, gt)
    recall_score = recall(binary_pred, gt)
    f1_score = f1(binary_pred, gt)

    # Store metrics in a dictionary
    metrics = {
        'builtup_accuracy': accuracy_score.item(),
        'builtup_precision': precision_score.item(),
        'builtup_recall': recall_score.item(),
        'builtup_f1': f1_score.item()
    }