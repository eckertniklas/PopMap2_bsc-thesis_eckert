import torch.nn.functional as F
import torch


def get_loss(output, sample):
    # y, mask_hr, mask_lr = (sample[k] for k in ('y', 'mask_hr', 'mask_lr'))
    y_pred = output["Popcount"]

    l1_loss = F.l1_loss(y_pred, sample["y"])
    log_l1_loss = F.l1_loss(torch.log(y_pred+1), torch.log(sample["y"]+1))
    mse_loss = F.mse_loss(y_pred, sample["y"])
    log_mse_loss = F.mse_loss(torch.log(y_pred+1), torch.log(sample["y"]+1))
    r2 = r2_loss(y_pred, sample["y"])
    mape = mape_func(y_pred, sample["y"])


    loss = l1_loss

    return loss, {
        'l1_loss': l1_loss.detach().item(),
        'log_l1_loss': log_l1_loss.detach().item(),
        'mse_loss': mse_loss.detach().item(),
        'log_mse_loss': log_mse_loss.detach().item(),
        'optimization_loss': loss.detach().item(),
        'r2': r2.detach().item(),
        'mape': mape.detach().item()
    }
        

# def mse_loss_func(pred, gt, mask):
#     return F.mse_loss(pred[mask == 1.], gt[mask == 1.])


# def l1_loss_func(pred, gt, mask):
#     return F.l1_loss(pred[mask == 1.], gt[mask == 1.])

def mape_func(pred, gt, eps=1e-8):
    pos_mask = gt>0.1
    return ( (pred[pos_mask]- gt[pos_mask]).abs() / (gt[pos_mask] + eps) - 1).mean()

# https://stackoverflow.com/questions/65840698/how-to-make-r2-score-in-nn-lstm-pytorch
def r2_loss(pred, gt):
    gt_mean = torch.mean(gt)
    ss_tot = torch.sum((gt - gt_mean) ** 2)
    ss_res = torch.sum((gt - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2