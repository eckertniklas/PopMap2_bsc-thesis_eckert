# https://github.com/fungtion/DANN

from torch.autograd import Function
from torch import nn
import torch.nn.functional as F

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    

# define the Domain Classifier
class DomainClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier, self).__init__()

        self.domain_classifier1 = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.domain_classifier2 = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.domain_classifier1(input_data)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = self.domain_classifier2(x.view(x.size(0), -1)).view(-1)
        return x
    

# define the Domain Classifier
class DomainClassifier1x1(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier1x1, self).__init__()

        self.domain_classifier1 = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=1, padding=0), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.ReLU(),
        )
        self.domain_classifier2 = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.domain_classifier1(input_data)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = self.domain_classifier2(x.view(x.size(0), -1)).view(-1)
        return x
    
# define the Domain Classifier
class DomainClassifier_v3(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier_v3, self).__init__()

        self.domain_classifier1 = nn.Sequential(
            nn.Conv2d(feature_dim, 100, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(100, 100, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.domain_classifier2 = nn.Sequential(
            nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.domain_classifier1(input_data)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = self.domain_classifier2(x.view(x.size(0), -1)).view(-1)
        return x
    

# define the Domain Classifier
class DomainClassifier_v4(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier_v4, self).__init__()

        self.domain_classifier1 = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=1, padding=1), nn.ReLU(),
        )
        self.domain_classifier2 = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.domain_classifier1(input_data)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = self.domain_classifier2(x.view(x.size(0), -1)).view(-1)
        return x
    
# define the Domain Classifier
class DomainClassifier_v5(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier_v5, self).__init__()

        self.domain_classifier1 = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=1, padding=0), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.domain_classifier1(input_data)
        return x