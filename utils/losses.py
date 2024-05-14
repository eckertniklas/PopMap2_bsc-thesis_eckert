import torch.nn.functional as F
import torch

from utils.plot import plot_2dmatrix
from collections import defaultdict


def get_loss(output, gt, scale=None,
             loss=["l1_loss"], lam=[1.0],
             builtuploss=False, lam_bul = [0.5], basicmethod=False, twoheadmethod=False,
             tag="",
             scale_regularization=0.0,
             ):
    """
    Compute the loss for the model
    input:
        output: dict of model outputs
        gt: dict of ground truth
        loss: list of losses to be used
        lam: list of weights for each loss
        lam_mmd: float, weight for the MMD loss
        builtuploss: bool, activates the builtup-loss add-on
        lam_bul: float, weight for the builtup-loss
        tag: str, tag to be used for logging
        scale_regularization: float, weight for the scale regularization
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

    if builtuploss == True:
        if output["builtup_score"].dtype != torch.float32:
            output["builtup_score"] = output["builtup_score"].float()
    
    # prepare vars1.0
    y_pred = output["popcount"]
    y_gt = gt["y"]
    if "popvar" in output.keys():
        if output["popvar"].dtype != torch.float32:
            output["popvar"] = output["popvar"].float()
        var = output["popvar"]

    # Population loss and metrics
    metricdict = {
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

    if twoheadmethod:
        metricdict["builtuploss"] = torch.nn.functional.binary_cross_entropy(input=output["twohead_builtup_score"], target=gt["building_segmentation"])
    if basicmethod:
        metricdict["builtuploss"] = torch.nn.functional.binary_cross_entropy(input=output["builtup_score"], target=gt["building_segmentation"])

    # define optimization loss as a weighted sum of the losses
    optimization_loss = torch.tensor(0, device=y_pred.device, dtype=y_pred.dtype)
    for lo,la in zip(loss,lam):
        if lo in metricdict.keys():
            optimization_loss += metricdict[lo] * la

    # occupancy scale regularization
    if scale is not None:
        if torch.isnan(scale).any():
            print("NaN values detected in scale.")
        if torch.isinf(scale).any():
            print("inf values detected in scale.")
        metricdict["scale"] = scale.float().abs().mean()
        # popdict["scaleL2"] = scale.float().pow(2).mean()
        if scale_regularization>0.0:
            optimization_loss += scale_regularization * metricdict["scale"]
        # if scale_regularizationL2>0.0:
        #     optimization_loss += scale_regularizationL2 * popdict["scaleL2"]

    # prepare for logging
    if tag=="":
        auxdict = {**auxdict, **{"Population"+"/"+key: value for key,value in metricdict.items()}}
    else:
        auxdict = {**auxdict, **{"Population_"+tag+"/"+key: value for key,value in metricdict.items()}}

    # log optimloss without builtup for debugging
    #FIXME: logging problems
    if builtuploss:
        auxdict["optimization_loss_basic"] =  optimization_loss

    # call builtup-loss function
    if builtuploss and basicmethod:
        bu_loss = torch.nn.functional.binary_cross_entropy(input=output["builtup_score"], target=gt["building_segmentation"])
        # bu_loss = builtup_lossfunction(output["builtup_score"], gt["building_segmentation"], lam_bul)
        # add builtuploss to optimization_loss
        optimization_loss = (1-lam_bul)*optimization_loss + lam_bul*bu_loss

    if builtuploss and twoheadmethod:
        bu_loss = torch.nn.functional.binary_cross_entropy(input=output["twohead_builtup_score"], target=gt["building_segmentation"])
        # bu_loss = builtup_lossfunction(output["twohead_builtup_score"], gt["building_segmentation"], lam_bul)
        # add builtuploss to optimization_loss
        optimization_loss = (1-lam_bul)*optimization_loss + lam_bul*bu_loss

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
    --------
    pred : tensor
        The predicted values.
    gt : tensor
        Ground truth values.

    Returns
    --------
    r2 : tensor
        The R2 score.

    Formula
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


def builtup_lossfunction(builtupscore, building_segmentation, lam=1.0):
    """
    BCELoss - pytorch
    """

    lossfunction = torch.nn.BCELoss(weight=torch.tensor(lam, device="cuda"))

    loss = lossfunction(builtupscore, building_segmentation)

    return loss