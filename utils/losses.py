import torch.nn.functional as F
import torch

from utils.plot import plot_2dmatrix
from collections import defaultdict
from utils.CORAL import coral
from utils.MMD import default_mmd as mmd

from torch.nn.modules.loss import _Loss
from torch import Tensor


def get_loss(output, gt, scale=None, loss=["l1_loss"], lam=[1.0], merge_aug=False,
             lam_adv=0.0, lam_coral=0.0, lam_mmd=0.0, tag="", scale_regularization=0.0):
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
        lam_coral: float, weight for the coral loss
        lam_mmd: float, weight for the mmd loss
    output:
        loss: float, the loss
        auxdict: dict, auxiliary losses
    """
    auxdict = defaultdict(float)
    
    # prepare vars1.0
    y_pred = output["popcount"][gt["source"]]
    y_gt = gt["y"][gt["source"]]
    if "popvar" in output.keys():
        var = output["popvar"][gt["source"]]

    # Population loss and metrics
    popdict = {
        "l1_loss": F.l1_loss(y_pred, y_gt),
        "log_l1_loss": F.l1_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mse_loss": F.mse_loss(y_pred, y_gt),
        "log_mse_loss": F.mse_loss(torch.log(y_pred+1), torch.log(y_gt+1)),
        "mr2": r2(y_pred, y_gt) if len(y_pred)>1 else torch.tensor(0.0),
        "mape": mape_func(y_pred, y_gt),
        # "focal_loss": focal_loss(y_pred, y_gt),
        # "tversky_loss": tversky_loss(y_pred, y_gt),
        "GTmean": y_gt.mean(),
        "GTstd": y_gt.std(),
        "predmean": y_pred.mean(),
        "predstd": y_pred.std(),
        "mCorrelation": torch.corrcoef(torch.stack([y_pred, y_gt]))[0,1] if len(y_pred)>1 else torch.tensor(0.0),
    }

    if "admin_mask" in gt.keys():
        popdict["L1reg"] = (output["popdensemap"] * (gt["admin_mask"]==gt["census_idx"].view(-1,1,1))).abs().mean()
    else:
        # popdict["L1reg"] = torch.abs(output["popdensemap"]).mean()
        popdict["L1reg"] = output["popdensemap"].abs().mean()

    if "popvar" in output.keys():
        varpopdict = {
            "gaussian_nll": gaussian_nll(y_pred, y_gt, var),
            "log_gaussian_nll": gaussian_nll(torch.log(y_pred+1), torch.log(y_gt+1), var),
            "laplacian_nll": laplacian_nll(y_pred, y_gt, var),
            "log_laplacian_nll": laplacian_nll(torch.log(y_pred+1), torch.log(y_gt+1), var),
            "STDpredmean": var.sqrt().mean(),
            # "log_gaussian_nll": gaussian_nll(torch.log(y_pred+1), torch.log(y_gt+1)),
        }
        popdict = {**popdict, **varpopdict}

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
        
        if "popvar" in output.keys():
            popdict["gaussian_aug_loss"] = gaussian_nll(aug_pred, aug_gt, aug_var)
            popdict["log_gaussian_aug_loss"] = gaussian_nll(torch.log(aug_pred+1), torch.log(aug_gt+1), aug_var)
            popdict["laplacian_aug_loss"] = laplacian_nll(aug_pred, aug_gt, aug_var)
            popdict["log_laplacian_aug_loss"] = laplacian_nll(torch.log(aug_pred+1), torch.log(aug_gt+1), aug_var)
    else:
        popdict["l1_aug_loss"] = popdict["l1_loss"]*4

    # define optimization loss as a weighted sum of the losses
    optimization_loss = torch.tensor(0, device=y_pred.device, dtype=y_pred.dtype)
    for lo,la in zip(loss,lam):
        if lo in popdict.keys():
            optimization_loss += popdict[lo]*la
    # optimization_loss = sum([popdict[lo]*la for lo,la in zip(loss,lam)])

    # occupancy scale regularization
    if scale is not None:
        popdict = {**popdict, **{"scale": scale.abs().mean()}}
        if scale_regularization>0.0:
            optimization_loss += scale_regularization * popdict["scale"]

    # prepare for logging
    if tag=="":
        auxdict = {**auxdict, **{"Population"+"/"+key: value for key,value in popdict.items()}}
    else:
        auxdict = {**auxdict, **{"Population_"+tag+"/"+key: value for key,value in popdict.items()}}

    # Domain adaption losses
    if ~gt["source"].all():

        # Adversarial Domain adaptation loss
        if lam_adv>0.0 and output["domain"] is not None:
            # prepare vars
            if len(output["domain"].shape)==4:
                dims = output["domain"].shape
                pred_domain = output["domain"][:,0].reshape(-1)
                gt_domain = gt["source"].float().repeat(dims[-1]*dims[-2]).reshape(-1)
            if len(output["domain"].shape)==2:
                num_subsamples = output["domain"].size(1)
                pred_domain = output["domain"].view(-1)
                gt_domain = gt["source"].float().unsqueeze(1).repeat(1,num_subsamples).view(-1)
            else:
                pred_domain = output["domain"]
                gt_domain = gt["source"].float()

            # calculate loss
            adv_dict = {"bce": F.binary_cross_entropy(pred_domain, gt_domain )}
            optimization_loss += lam_adv*adv_dict["bce"]

            # prepate for logging
            adv_dict.update(**class_metrics(pred_domain, gt_domain, thresh=0.5))
            auxdict = {**auxdict, **{"Domainadaptation/adv/"+key: value for key,value in adv_dict.items()}}
        
        # CORAL Domain adaptation loss
        if lam_coral>0.0 and output["decoder_features"] is not None:
            source_features = output["decoder_features"][gt["source"]].permute(0,2,1).reshape(output["decoder_features"].shape[1],-1)
            target_features = output["decoder_features"][~gt["source"]].permute(0,2,1).reshape(output["decoder_features"].shape[1],-1)
            coral_dict = {"coral_loss": coral(source_features.T, target_features.T)}
            optimization_loss += lam_coral*coral_dict["coral_loss"]

            # prepate for logging
            auxdict = {**auxdict, **{"Domainadaptation/"+key: value for key,value in coral_dict.items()}}

        # MMD Domain adaptation loss
        if lam_mmd>0.0 and output["decoder_features"] is not None:
            source_features = output["decoder_features"][gt["source"]].permute(0,2,1).reshape(output["decoder_features"].shape[1],-1)
            target_features = output["decoder_features"][~gt["source"]].permute(0,2,1).reshape(output["decoder_features"].shape[1],-1)
            mmd_dict = {"mmd_loss": mmd(source_features.T, target_features.T)}
            optimization_loss += lam_mmd*mmd_dict["mmd_loss"]

            # prepate for logging
            auxdict = {**auxdict, **{"Domainadaptation/"+key: value for key,value in mmd_dict.items()}}
        
    # prepare for logging
    auxdict["optimization_loss"] =  optimization_loss
    auxdict = {key:value.detach().item() for key,value in auxdict.items()}

    return optimization_loss, auxdict
                         

class LogL1Loss(_Loss):

    def __init__(self, *, full: bool = False, reduction: str = 'mean') -> None:
        super(LogL1Loss, self).__init__(None, None, reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(torch.log(pred+1), torch.log(target+1))
    

class LaplacianNLLLoss(_Loss):
    """Laplacian negative log-likelihood loss using log-variance

    For more details see:
        I don't know yet

    Adapted from:
        torch.nn.GaussianNLLLoss: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss
        torch.nn.functional.gaussian_nll_loss: https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
    """
    __constants__ = ['full', 'eps', 'max_clamp', 'reduction']
    full: bool
    eps: float
    max_clamp: float

    def __init__(self, *, full: bool = False, max_clamp: float = 10.0, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(LaplacianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.max_clamp = max_clamp # for exponential stability 
        self.eps = eps # for division stability
        self.eps2 = 1e-3

        if self.full:
            print('FULL LAPLACIAN NLL LOSS NOT YET IMPLEMENTED')
            raise NotImplementedError

    def forward(self, pred: Tensor, var:Tensor, target: Tensor) -> Tensor:
        # pred, var = pred_log_var.split([1,1], dim=0)
        log_var = (var+self.eps2).log()
        loss =  0.5 * (log_var + abs(pred - target) / (var + self.eps))
        # loss =  0.5 * (log_var + abs(pred - target) / (torch.exp(log_var.clamp(max=self.max_clamp)) + self.eps))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

BCE = torch.nn.BCELoss()
gaussian_nll =  torch.nn.GaussianNLLLoss()
laplacian_nll = LaplacianNLLLoss()



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