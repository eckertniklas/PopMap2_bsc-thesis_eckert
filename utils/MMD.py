
import torch
from torch import nn

import pytest
# from utils.utils import plot_2dmatrix

# from https://github.com/yiftachbeer/mmd_loss_pytorch
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        """
        :param n_kernels: number of kernels
        :param mul_factor: factor by which the bandwidth of each kernel is multiplied
        :param bandwidth: bandwidth of the RBF kernel
        """
        super().__init__()


        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels, device="cuda" if torch.cuda.is_available else "cpu") - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """
        Compute the bandwidth of the RBF kernel
        :param L2_distances: L2 distances between all pairs of points
        :return: bandwidth of the RBF kernel
        """
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        """
        Compute the RBF kernel between all pairs of points in X
        :param X: input data (N * d)
        :return: RBF kernel (N * N)
        """
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        """
        :param kernel: kernel function
        """
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        """
        Compute the MMD loss between X and Y
        :param X: input data (N * d)
        :param Y: input data (N * d)
        :return: MMD loss (scalar)
        """
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    

default_mmd = MMDLoss()



if __name__ == "__main__":

    from utils import plot_2dmatrix
        
    @pytest.fixture
    def set_seed(seed=123):
        torch.manual_seed(seed)
        yield
        torch.manual_seed(torch.initial_seed())

    def test_mmd_loss(set_seed):
        # Test case 1: Test that the MMD loss is 0 when X and Y are the same
        X = torch.randn(16, 7)
        Y = X.clone()
        mmd_loss = MMDLoss()
        assert mmd_loss(X, Y) == 0.0

        # Test case 2: Test that the MMD loss is greater than 0 when X and Y are different
        X = torch.randn(16, 7)
        Y = torch.randn(16, 7)
        mmd_loss = MMDLoss()
        assert mmd_loss(X, Y) > 0.0

        # # Test case 3: Test that the MMD loss is 0 when X and Y have the same mean
        # X = torch.randn(16, 7)
        # Y = torch.randn(16, 7)
        # Y = Y - Y.mean() + X.mean()
        # mmd_loss = MMDLoss()
        # assert mmd_loss(X, Y) == 0.0

        # Test case 4: Test that the MMD loss is not 0 when X and Y have different means
        X = torch.randn(16, 7)
        Y = X + torch.randn(1, 7) * 0.1
        mmd_loss = MMDLoss()
        assert mmd_loss(X, Y) > 0.0

        # Test case 5: Test that the MMD loss is equal to the squared L2 distance between X and Y when using a linear kernel
        # X = torch.randn(16, 7)
        # Y = X + 0.1
        # linear_kernel = nn.Linear(32, 7)
        # mmd_loss = MMDLoss(kernel=linear_kernel)
        # assert mmd_loss(X, Y) == torch.sum((X.mean(dim=0) - Y.mean(dim=0)) ** 2)

        # Test case 6: Test that the MMD loss is equal to the squared L2 distance between X and Y when using an RBF kernel with a large bandwidth
        # X = torch.randn(10, 7)
        # Y = X + 0.1
        # rbf_kernel = RBF(bandwidth=100)
        # mmd_loss = MMDLoss(kernel=rbf_kernel)
        # assert mmd_loss(X, Y) == torch.sum((X.mean(dim=0) - Y.mean(dim=0)) ** 2)

    # Run the tests
    test_mmd_loss(1616)
    print("All tests passed!")