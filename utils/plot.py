import os
import numpy as np
import torch
from pylab import figure, imshow, matshow, grid, savefig, colorbar
import matplotlib.pyplot as plt



def plot_2dmatrix(matrix, fig=1, vmin=None, vmax=None):
    if torch.is_tensor(matrix):
        if matrix.is_cuda:
            matrix = matrix.cpu()
        if matrix.requires_grad:
            matrix = matrix.detach()
        matrix = matrix.numpy()
    if matrix.shape[0]==1:
        matrix = matrix[0]
    if matrix.shape[0]==3:
        matrix = matrix.transpose((1,2,0))

    figure(fig)
    matshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
    grid(True)
    colorbar()
    savefig('plot_outputs/last_plot.png')



def plot_and_save(img, mask=None, vmax=None, vmin=None, idx=None,
    model_name='model_name', title=None, name='latest_figure', colorbar=True, cmap="viridis", folder='vis'):
    """
    :param img: image to plot
    :param mask: mask to apply to image
    :param vmax: max value for colorbar
    :param vmin: min value for colorbar
    :param idx: index of image
    :param model_name: name of model
    :param title: title of plot
    :param name: name of plot
    :param colorbar: whether to plot colorbar or not
    :param cmap: colormap
    :param folder: folder to save plot
    """

    folder = os.path.join(folder, "vis")

    if mask is not None:
        img = np.ma.masked_where(mask, img)
    
    plt.figure(figsize=(12, 8), dpi=260)
    if vmax is not None or vmin is not None:
        img = np.clip(img, vmin, vmax)
    if img.dim()==3:
        if img.shape[2]==3:
            img = np.clip(img, 0.000000001, 0.999999999)

    plt.imshow(img, vmax=vmax, vmin=vmin, cmap=cmap)
    if colorbar:
        plt.colorbar()
    title = title if title is not None else model_name
    plt.title(title)

    if idx is not None:
        if not os.path.exists(f'{folder}/{model_name}/'):
            os.makedirs(f'{folder}/{model_name}/')
        plt.savefig(f'{folder}/{model_name}/{str(idx).zfill(4)}_{name}.png')
        plt.close()
    else:
        plt.savefig(f'{name}.png')
