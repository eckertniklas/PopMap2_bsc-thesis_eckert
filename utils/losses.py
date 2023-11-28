import torch.nn.functional as F
import torch

from utils.plot import plot_2dmatrix
from collections import defaultdict
# from utils.CORAL import coral
# from utils.MMD import default_mmd as mmd

from torch.nn.modules.loss import _Loss
from torch import Tensor


def get_loss(output, gt, scale=None, empty_scale=None, loss=["l1_loss"], lam=[1.0], merge_aug=False,
             tag="",
             scale_regularization=0.0, scale_regularizationL2=0.0, emptyscale_regularizationL2=0.0,
             output_regularization=0.0):
    """
    Compute the loss for the model
    input:
        output: dict of model outputs
        gt: dict of ground truth
        loss: list of losses to be used
        lam: list of weights for each loss
        merge_aug: bool, if True, merge the losses to create fake administrative territories
        lam_builtmask: float, weight for the built mask loss
        lam_adv: float, weight for the adversarial loss
        lam_coral: float, weight for the CORAL loss
        lam_mmd: float, weight for the MMD loss
        tag: str, tag to be used for logging
        scale_regularization: float, weight for the scale regularization
        scale_regularizationL2: float, weight for the L2 scale regularization
        output_regularization: float, weight for the output regularization
    output:
        loss: float, the loss
        auxdict: dict, auxiliary losses
    """
    auxdict = defaultdict(float)

    # check that all tensors are float32
    if output["popcount"].dtype != torch.float32:
        output["popcount"] = output["popcount"].float()
    
    if output["popdensemap"].dtype != torch.float32:
        output["popdensemap"] = output["popdensemap"].float()

    if output["scale"] is not None:
        if output["scale"].dtype != torch.float32:
            output["scale"] = output["scale"].float()
    
    if output["empty_scale"] is not None:
        if output["empty_scale"].dtype != torch.float32:
            output["empty_scale"] = output["empty_scale"].float()
        
    # prepare vars1.0
    y_pred = output["popcount"][gt["source"]]
    y_gt = gt["y"][gt["source"]]
    if "popvar" in output.keys():
        if output["popvar"].dtype != torch.float32:
            output["popvar"] = output["popvar"].float()
        var = output["popvar"][gt["source"]]

    # Population loss and metrics
    popdict = {
        "l1_loss": F.l1_loss(y_pred, y_gt),
        "log_l1_loss": F.l1_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mse_loss": F.mse_loss(y_pred, y_gt),
        "log_mse_loss": F.mse_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mr2": r2(y_pred, y_gt) if len(y_pred)>1 else torch.tensor(0.0),
        "mape": mape_func(y_pred, y_gt),
        "GTmean": y_gt.mean(),
        "GTstd": y_gt.std(),
        "predmean": y_pred.mean(),
        "predstd": y_pred.std(),
        "mCorrelation": torch.corrcoef(torch.stack([y_pred, y_gt]))[0,1] if len(y_pred)>1 else torch.tensor(0.0),
    }

    if "admin_mask" in gt.keys():
        popdict["L1reg"] = (output["popdensemap"] * (gt["admin_mask"]==gt["census_idx"].view(-1,1,1))).abs().mean()
    else:
        popdict["L1reg"] = output["popdensemap"].abs().mean()

    # augmented loss
    if len(y_pred)%merge_aug==0:
        hl = len(y_pred)//merge_aug
        aug_pred = torch.stack(torch.split(y_pred, hl)).sum(0) / merge_aug * 2
        if "popvar" in output.keys():
            aug_var = torch.stack(torch.split(var, hl)).sum(0) / merge_aug * 2
        aug_gt = torch.stack(torch.split(y_gt, hl)).sum(0) / merge_aug * 2
        popdict["l1_aug_loss"] = F.l1_loss(aug_pred, aug_gt)
        popdict["log_l1_aug_loss"] = F.l1_loss(torch.log(aug_pred+1), torch.log(aug_gt+1))
        popdict["mse_aug_loss"] = F.mse_loss(aug_pred, aug_gt)
        popdict["log_mse_aug_loss"] = F.mse_loss(torch.log(aug_pred+1), torch.log(aug_gt+1))
        
    else:
        popdict["l1_aug_loss"] = popdict["l1_loss"]*4

    # define optimization loss as a weighted sum of the losses
    optimization_loss = torch.tensor(0, device=y_pred.device, dtype=y_pred.dtype)
    for lo,la in zip(loss,lam):
        if lo in popdict.keys():
            optimization_loss += popdict[lo] * la

    # occupancy scale regularization
    if scale is not None:
        if torch.isnan(scale).any():
            print("NaN values detected in scale.")
        if torch.isinf(scale).any():
            print("inf values detected in scale.")
        popdict["scale"] = scale.float().abs().mean()
        popdict["scaleL2"] = scale.float().pow(2).mean()
        if scale_regularization>0.0:
            optimization_loss += scale_regularization * popdict["scale"]
        if scale_regularizationL2>0.0:
            optimization_loss += scale_regularizationL2 * popdict["scaleL2"]

    # empty scale regularization
    if empty_scale is not None:
        popdict["empty_scale"] = empty_scale.float().abs().mean()
        popdict["empty_scaleL2"] = empty_scale.float().pow(2).mean()
        if emptyscale_regularizationL2>0.0:
            optimization_loss += emptyscale_regularizationL2 * popdict["empty_scaleL2"]

    if output_regularization>0.0:
        optimization_loss += output_regularization * output["popcount"].abs().mean()

    # prepare for logging
    if tag=="":
        auxdict = {**auxdict, **{"Population"+"/"+key: value for key,value in popdict.items()}}
    else:
        auxdict = {**auxdict, **{"Population_"+tag+"/"+key: value for key,value in popdict.items()}}

    # prepare for logging
    auxdict["optimization_loss"] =  optimization_loss
    auxdict = {key:value.detach().item() for key,value in auxdict.items()}

    return optimization_loss, auxdict
                         
def mape_func(pred, gt, eps=1e-8):
    """
    Calculate the mean absolute percentage error between the ground truth and the prediction.
    """
    pos_mask = gt>0.1
    mre =  ( (pred[pos_mask]- gt[pos_mask]).abs() / (gt[pos_mask] + eps)).mean()
    return mre*100


# adapted from https://stackoverflow.com/questions/65840698/how-to-make-r2-score-in-nn-lstm-pytorch
def r2(pred, gt, eps=1e-8):
    """
    Calculate the R2 score between the ground truth and the prediction.
    
    Parameters
    ----------
    pred : tensor
        The predicted values.
    gt : tensor
        Ground truth values.

    Returns
    -------
    r2 : tensor
        The R2 score.

    Forumula
    --------
    R2 = 1 - SS_res / SS_tot
    SS_res = sum((gt - pred) ** 2)
    SS_tot = sum((gt - gt_mean) ** 2)
    """
    gt_mean = torch.mean(gt)
    ss_tot = torch.sum((gt - gt_mean) ** 2)
    ss_res = torch.sum((gt - pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)
    return r2
