import torch
from torch import Tensor
# from utils.plot import plot_2dmatrix, plot_and_save

def coral(source: Tensor, target: Tensor) -> Tensor:
    """
    CORAL loss function
    :param source: source feature matrix (N * d)
    :param target: target feature matrix (N * d)
    :return: CORAL loss (scalar)
    """
    d = source.size(1)
    source_c = compute_covariance(source)
    target_c = compute_covariance(target)
    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
    loss = loss / (4 * d * d)
    return loss

def compute_covariance(input_data: Tensor) -> Tensor:
    """
    Compute covariance matrix of input data
    :param input_data: input data (N * d)
    :return: covariance matrix (d * d)
    """
    n = input_data.size(0)
    id_row = torch.ones(n).view(1, n).to(device=input_data.device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
    return c


if __name__=="__main__":

    from utils.plot import plot_2dmatrix
    
    # Test case
    source_features = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 1.0, 0.5, 1.5, 2.5], [0.0, 2.0, 1.0, 3.0, 4.0]], requires_grad=True)
    target_features = torch.tensor([[2.0, 1.0, 4.0, 4.0, 5.0], [1.0, 3.0, 2.0, 4.0, 5.0], [4.0, 0.0, 1.5, 4.0, 3.0]], requires_grad=True)
    loss = coral(source_features, target_features)
    loss.backward()
    print("CORAL Loss:", loss.item())
    print("Gradient for source features:", source_features.grad)
    print("Gradient for target features:", target_features.grad)


