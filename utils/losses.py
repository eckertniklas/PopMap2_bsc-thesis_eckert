import torch.nn.functional as F
import torch

from utils.utils import plot_2dmatrix
from collections import defaultdict


def get_loss(output, gt, loss=["l1_loss"], lam=[1.0], merge_aug=False, lam_builtmask=1., lam_dense=1., lam_adv=1.):
    auxdict = defaultdict(float)

    # TODO: if domain adversarial loss is used, split the prediction and output into labeled and unlabeled
    # TODO: calculate the adversarial loss and add it to the optimization loss
    
    # prepare vars1.0
    y_pred = output["popcount"][gt["source"]]
    y_gt = gt["y"][gt["source"]]


    # Population loss and metrics
    popdict = {
        "l1_loss": F.l1_loss(y_pred, y_gt),
        "log_l1_loss": F.l1_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mse_loss": F.mse_loss(y_pred, y_gt),
        "log_mse_loss": F.mse_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mr2": r2(y_pred, y_gt),
        "mape": mape_func(y_pred, y_gt),
        "focal_loss": focal_loss(y_pred, y_gt),
        "tversky_loss": tversky_loss(y_pred, y_gt),
        "GTmean": y_gt.mean(),
        "GTstd": y_gt.std(),
        "predmean": y_pred.mean(),
        "predstd": y_pred.std(),
        "mCorrelation": torch.corrcoef(torch.stack([y_pred, y_gt]))[0,1]
    }

    # augmented loss
    if len(y_pred)%2==0:
        hl = len(y_pred)//2
        aug_pred = torch.stack(torch.split(y_pred, hl)).sum(0)
        aug_gt = torch.stack(torch.split(y_gt, hl)).sum(0) 
        popdict["l1_aug_loss"] = F.l1_loss(aug_pred, aug_gt)
        popdict["mse_aug_loss"] = F.mse_loss(aug_pred, aug_gt)
    else:
        popdict["l1_aug_loss"] = popdict["l1_loss"]*4

    # define optimization loss
    if merge_aug:
        optimization_loss = popdict["mse_aug_loss"]
    else:
        # optimization_loss = popdict[loss]
        optimization_loss = sum([popdict[lo]*la for lo,la in zip(loss,lam)])

    popdict = {"Population/"+key: value for key,value in popdict.items()}
    auxdict = {**auxdict, **popdict}

    # Adversarial loss
    if ~gt["source"].all():
        adv_dict = {}

        if len(output["domain"].shape)==4:
            dims = output["domain"].shape
            pred_domain = output["domain"][:,0].reshape(-1)
            gt_domain = gt["source"].float().repeat(dims[-1]*dims[-2]).reshape(-1)
        else:
            pred_domain = output["domain"]
            gt_domain = gt["source"].float()

        bce = F.binary_cross_entropy(pred_domain, gt_domain )
        optimization_loss += lam_adv*bce

        # prepate for logging
        adv_dict["bce"] = bce
        adv_dict.update(**class_metrics(pred_domain, gt_domain, thresh=0.5))
        adv_dict = {"Adversarial/"+key: value for key,value in adv_dict.items()}
        auxdict = {**auxdict, **adv_dict}
    
    # Builtup mask loss
    disabled = True
    if "builtupmap" in gt and not disabled:
        y_bpred = output["builtupmap"] 

        builtupdict = {
            **{
                "bce": BCE(output["builtupmap"], gt["builtupmap"]),
                "focal_loss": focal_loss(output["builtupmap"].view(-1), gt["builtupmap"].view(-1)),
                "tversky_loss": tversky_loss(output["builtupmap"].view(-1), gt["builtupmap"].view(-1))
            },
            **class_metrics(output["builtupmap"], gt["builtupmap"], thresh=0.5)
        }

        if lam_builtmask>0.0:
            optimization_loss += lam_builtmask*builtupdict["bce"]

        # Building density calculation
        builtdensedict = {}

        builtupdict = {"builtup:"+key: value for key,value in builtupdict.items()}
        builtdensedict = {"builtdense:"+key: value for key,value in builtdensedict.items()}
        auxdict = {**auxdict, **builtupdict}
        auxdict = {**auxdict, **builtdensedict}
        
    auxdict["optimization_loss"] =  optimization_loss
    auxdict = {key:value.detach().item() for key,value in auxdict.items()}

    return optimization_loss, auxdict

        
BCE = torch.nn.BCELoss()


def class_metrics(pred, gt, thresh=0.5, eps=1e-8):
    pred = (pred.reshape(-1)>thresh).float()
    gt = gt.float().view(-1)

    # Get confusion matrix
    cm =  torch.stack([gt,1-gt],0) @ torch.stack([pred,1-pred],1)
    TP, FN = cm[0,0], cm[0,1]
    FP, TN = cm[1,0], cm[1,1]

    # get metrics
    acc = (TP + TN) / (cm.sum() + eps)
    P = TP / (TP + FP + eps)
    R = TP / (TP + FN + eps)
    F1 = 2 * (P*R) / (P + R + eps)
    IoU = TP / (TP + FN + FP + eps)

    return {"accuracy": acc, "precision": P, "recall": R, "f1": F1, "IoU": IoU}


def mape_func(pred, gt, eps=1e-8):
    pos_mask = gt>0.1
    mre =  ( (pred[pos_mask]- gt[pos_mask]).abs() / (gt[pos_mask] + eps)).mean()
    return mre*100


# stolen from https://stackoverflow.com/questions/65840698/how-to-make-r2-score-in-nn-lstm-pytorch
def r2(pred, gt, eps=1e-8):
    gt_mean = torch.mean(gt)
    ss_tot = torch.sum((gt - gt_mean) ** 2)
    ss_res = torch.sum((gt - pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)
    return r2



def negative_binomial_loss(y_true, y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.
        
    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    # Separate the parameters
    n, p = torch.unbind(y_pred, dim=1)
    
    # # Add one dimension to make the right shape
    # n = tf.expand_dims(n, -1)
    # p = tf.expand_dims(p, -1)
    
    # Calculate the negative log likelihood
    nll = (
        torch.lgamma(n) 
        + torch.lgamma(y_true + 1)
        - torch.lgamma(n + y_true)
        - n * torch.log(p)
        - y_true * torch.log(1 - p)
    )                  

    return nll





























def focal_loss(pred, gt, alpha=0.5, gamma=2):
    gt = torch.stack([gt, 1-gt],1)
    pred = torch.stack([pred, 1-pred],1)
    
    weight = torch.pow(1. - pred, gamma)
    focal = -alpha * weight * torch.log(pred)
    return torch.sum(gt * focal, dim=1).mean()
    # return torch.sum(gt * focal, dim=1)
    

def tversky_loss(pred, gt, eps=1e-8, alpha=0.5, beta=0.5, dims=(1)):
    gt = torch.stack([gt, 1-gt],1)
    pred = torch.stack([pred, 1-pred],1)

    intersection = torch.sum(pred * gt, dims)
    fps = torch.sum(pred * (1. - gt), dims)
    fns = torch.sum((1. - pred) * gt, dims)

    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + eps)
    return torch.mean(1. - tversky_loss)