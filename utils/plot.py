import os
import numpy as np
import torch
from pylab import figure, imshow, matshow, grid, savefig, colorbar
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_2dmatrix(matrix, fig=1, vmin=None, vmax=None, show=False):
    if torch.is_tensor(matrix):
        if matrix.is_cuda:
            matrix = matrix.cpu()
        if matrix.requires_grad:
            matrix = matrix.detach()
        matrix = matrix.numpy()
    if matrix.shape[0]==1 and len(matrix.shape)==3:
        matrix = matrix[0]
    if matrix.shape[0]==3 and len(matrix.shape)==3:
        matrix = matrix.transpose((1,2,0))

    plt.figure(fig)
    plt.matshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.grid(True)
    plt.colorbar()
    plt.savefig('plot_outputs/last_plot.png')
    if show: 
        plt.show()


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
    print("min_value, max_value", min_value, max_value)
    
    # I want to know the min and max, but in terms of perceptage of the total
    min_value_perc = min_value / len(predicted)
    max_value_perc = max_value / len(predicted)
    print("min_value_perc, max_value_perc", min_value_perc, max_value_perc)
    print("min [%], max [%]", min_value_perc*100, max_value_perc*100)
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



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from io import BytesIO
from PIL import Image

def scatter_plot_with_zeros(predicted, ground_truth, log_scale=True):
    # Convert input data to numpy arrays
    x = np.array(predicted)
    y = np.array(ground_truth)

    # Create the main scatter plot for non-zero data
    fig, ax_main = plt.subplots()
    
    mask_non_zero = (x != 0) & (y != 0)
    x_non_zero = x[mask_non_zero]
    y_non_zero = y[mask_non_zero]
    
    if len(x_non_zero) > 2:
        # Calculate point density for color scale
        xy = np.vstack([x_non_zero, y_non_zero])
        z = gaussian_kde(xy)(xy)
        ax_main.scatter(x_non_zero, y_non_zero, c=z, s=12)

        if log_scale:
            ax_main.set_xscale('log')
            ax_main.set_yscale('log')
            ax_main.set_xlim(min(0.5, np.min(x_non_zero)), np.max(x_non_zero))
            ax_main.set_ylim(min(0.5, np.min(y_non_zero)), np.max(y_non_zero))

    # Create insets for zeros
    ax_x_inset = fig.add_axes([0.1, 0.1, 0.4, 0.03])  # [x, y, width, height]
    ax_x_inset.set_xticks([])
    ax_x_inset.set_yticks([])
    ax_x_inset.spines['top'].set_visible(False)
    ax_x_inset.spines['right'].set_visible(False)
    ax_x_inset.spines['bottom'].set_visible(False)
    ax_x_inset.spines['left'].set_visible(False)
    ax_x_inset.plot([0, 1], [0.5, 0.5], color='gray', linestyle='--')
    ax_x_inset.scatter(np.linspace(0, 1, np.sum(x == 0)), [0.5]*np.sum(x == 0), marker='|', color='blue')

    ax_y_inset = fig.add_axes([0.1, 0.15, 0.03, 0.4])
    ax_y_inset.set_xticks([])
    ax_y_inset.set_yticks([])
    ax_y_inset.spines['top'].set_visible(False)
    ax_y_inset.spines['right'].set_visible(False)
    ax_y_inset.spines['bottom'].set_visible(False)
    ax_y_inset.spines['left'].set_visible(False)
    ax_y_inset.plot([0.5, 0.5], [0, 1], color='gray', linestyle='--')
    ax_y_inset.scatter([0.5]*np.sum(y == 0), np.linspace(0, 1, np.sum(y == 0)), marker='_', color='red')

    # Set labels and title
    ax_main.set_xlabel('Predicted Values')
    ax_main.set_ylabel('Ground Truth Values')
    ax_main.set_title('Predicted vs. Ground Truth Values')
    
    return fig, ax_main



def scatter_plot_with_zeros_v9(predicted, ground_truth, log_scale=True):

    show = False
    if show:
        import matplotlib
        matplotlib.use('TkAgg')  # or 'Qt5Agg'
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

    # Convert the input lists to numpy arrays for efficient operations
    x = np.array(predicted)
    y = np.array(ground_truth)

    # Define the figure
    fig = plt.figure(figsize=(3.7, 3))

    # # Define a 3x3 grid
    gs = gridspec.GridSpec(10, 10)
    ax_big = plt.subplot(gs[0:9, 1:10])
    ax_slim_left = plt.subplot(gs[0:9, 0])
    ax_slim_bottom = plt.subplot(gs[9, 1:10])
    ax_zero = plt.subplot(gs[9, 0])

    # Create a 2D array combining x and y to calculate point density
    xy = np.vstack([x, y])
    # Apply log transformation to the data if log_scale is True
    if log_scale:
        xy = np.vstack([np.log10(x+0.1), np.log10(y+0.1)])

    # Calculate point density using Gaussian Kernel Density Estimation
    z = gaussian_kde(xy)(xy)
    
    # Create a mask to filter out zero values from x and y
    mask_non_zero = (x != 0) & (y != 0)
    x_non_zero = x[mask_non_zero]
    y_non_zero = y[mask_non_zero]
    z_non_zero = z[mask_non_zero]
    
    # filter for axis histogram
    has_y = True
    y1 = y[(x < 0.5)* (y > 0.5)] # only pred vals
    if len(y1) > 2:
        if log_scale:
            y11 = np.log10(y1)
        else:
            y11 = y1
        # y11x = np.vstack([np.zeros_like(y11),y11])
        # z11 = gaussian_kde(xy)(y11x)
        z11 = gaussian_kde(y11)(y11)
    else:
        z11 = np.array([0])
        y1 = np.array([0])
        has_y = False
    
    x1 = x[(x > 0.5)* (y < 0.5)] # only gt vals
    if len(x1) > 2:
        if log_scale:
            x11 = np.log10(x1)
        else:
            x11 = x1
        # x11y = np.vstack([x11, np.zeros_like(x11)])
        # z12 = gaussian_kde(xy)(x11y)
        if not has_y:
            z12 = gaussian_kde(x11)(x11)
        else:
            z12 = gaussian_kde(y11)(x11)
    else:
        z12 = np.array([0])
        x1 = np.array([0])

    mask_zero = (x < 0.5) & (y < 0.5)
    z_zero = z[mask_zero]

    # get global min and max for colorbar
    z_min = min(np.min(z_non_zero), np.min(z11), np.min(z12))
    z_max = max(np.max(z_non_zero), np.max(z11), np.max(z12))
        
    # Calculate the global min and max for both x and y
    global_min = 0.5
    global_max = max(np.max(x_non_zero), np.max(y_non_zero))

    # Plot non-zero x and y values if there are more than 2 points
    if len(x_non_zero) > 2:
        scatter_big = ax_big.scatter(x_non_zero, y_non_zero, c=z_non_zero, s=12, cmap='cividis') # Plot the scatter plot
        scatter_big.set_clim(vmin=0, vmax=np.max(z_non_zero))  # Set the color limits

        # Apply log scaling if log_scale is True
        if log_scale:
            ax_big.set_xscale('log')
            ax_big.set_yscale('log')

        ax_big.set_xlim(global_min, global_max)
        ax_big.set_ylim(global_min, global_max)

        # Remove the frame and labels from ax_big
        ax_big.axis('off')
        ax_big.spines['right'].set_visible(False)
        ax_big.spines['left'].set_visible(False)
        ax_big.spines['top'].set_visible(False)
        ax_big.spines['bottom'].set_visible(False)

        ax_big.set_xlabel('')
        ax_big.set_ylabel('')

        ax_big.plot([global_min, global_max], [global_min, global_max], color='red', linestyle='--', alpha=0.5)  # Add the identity line
        # ax_big.axline((1, 1), slope=1, color='red', linestyle='--', alpha=0.5)  # Add the identity line

    scatter_bottom = ax_slim_bottom.scatter(x1, np.zeros_like(x1), c=z12, s=12, cmap='cividis')
    scatter_bottom.set_clim(vmin=z_min, vmax=z_max)  # Set the color limits
    if log_scale:
        ax_slim_bottom.set_xscale('log')
    ax_slim_bottom.set_xlim(global_min, global_max)
    ax_slim_bottom.spines['right'].set_visible(False)
    ax_slim_bottom.spines['left'].set_visible(False)
    ax_slim_bottom.spines['top'].set_visible(False)
    # ax_slim_bottom.spines['bottom'].set_visible(False)
    ax_slim_bottom.set_yticks([])       # Removes x-axis ticks
    ax_slim_bottom.set_yticklabels([])  # Removes x-axis tick labels
    ax_slim_bottom.set_ylabel('')       # Removes x-axis label

    scatter_left = ax_slim_left.scatter(np.zeros_like(y1), y1, c=z11, s=12, cmap='cividis')
    scatter_left.set_clim(vmin=z_min, vmax=z_max)  # Set the color limits
    if log_scale:
        ax_slim_left.set_yscale('log')
    ax_slim_left.set_ylim(global_min, global_max)
    ax_slim_left.spines['right'].set_visible(False)
    # ax_slim_left.spines['left'].set_visible(False)
    ax_slim_left.spines['top'].set_visible(False)
    ax_slim_left.spines['bottom'].set_visible(False)
    ax_slim_left.set_xticks([])       # Removes x-axis ticks
    ax_slim_left.set_xticklabels([])  # Removes x-axis tick labels
    ax_slim_left.set_xlabel('')       # Removes x-axis label

    scatter_zero = ax_zero.scatter([0], [0], c=z_zero.sum(), s=12, cmap='cividis')
    scatter_zero.set_clim(vmin=z_min, vmax=z_max)  # Set the color limits
    ax_zero.set_xlim(-0.1, 0.1)
    ax_zero.set_ylim(-0.1, 0.1)
    ax_zero.set_yticks([0])
    ax_zero.set_yticklabels(['<0.5'])
    ax_zero.set_xticks([0])
    ax_zero.set_xticklabels(['<0.5'])
    ax_zero.spines['right'].set_visible(False)
    # ax_zero.spines['left'].set_visible(False)
    ax_zero.spines['top'].set_visible(False)
    # ax_zero.spines['bottom'].set_visible(False)

    ax_slim_bottom.set_xlabel('Predicted Values')
    ax_slim_left.set_ylabel('Ground Truth Values')
    plt.tight_layout()

    if show:
        plt.show()
    
    # return fig, ax_main
    return fig
