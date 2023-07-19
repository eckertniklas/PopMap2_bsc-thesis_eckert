import os
import numpy as np
import torch
from pylab import figure, imshow, matshow, grid, savefig, colorbar
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


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


def scatter_plot(predicted, ground_truth):
    # Create a scatterplot of the predicted and ground truth values
    plt.scatter(predicted, ground_truth)

    # Add axis labels and a title
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth Values')
    plt.title('Predicted vs. Ground Truth Values')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Open the BytesIO object as a PIL Image and return it
    return Image.open(buffer)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def scatter_plot2(predicted, ground_truth):
    # Create a scatterplot of the predicted and ground truth values
    
    tips = sns.load_dataset("tips")

    values = np.vstack([predicted, ground_truth])
    kernel = stats.gaussian_kde(values)(values)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=tips,
        x="total_bill",
        y="tip",
        c=kernel,
        cmap="viridis",
        ax=ax,
    )

    # Add axis labels and a title
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth Values')
    plt.title('Predicted vs. Ground Truth Values')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Open the BytesIO object as a PIL Image and return it
    return Image.open(buffer)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

def scatter_plot3(predicted, ground_truth, log_scale=True):
    # Create a scatterplot of the predicted and ground truth values

    x = np.array(predicted)
    y = np.array(ground_truth)

    # Remove zeros from x and y
    mask = (x != 0) & (y != 0)
    if mask.sum() <= 2:
        return None
    
    x = x[mask]
    y = y[mask]

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=12)

    # Set the x and y limits to match the ground truth min and max
    if log_scale:
        min_value, max_value = np.max([0.5,np.min(ground_truth)]), np.max(ground_truth)
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    # plt.show()

    # Set both axes to a log scale
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    # Add axis labels and a title
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth Values')
    plt.title('Predicted vs. Ground Truth Values')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # close the figure
    plt.close()

    # Open the BytesIO object as a PIL Image and return it
    return Image.open(buffer)

