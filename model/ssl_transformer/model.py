import torch
import torch.nn as nn

import torchvision.models as models

class DoubleResNetSimCLRDownstream(torch.nn.Module):
    """concatenate outputs from two backbones and add one linear layer"""

    def __init__(self, base_model, out_dim):
        super(DoubleResNetSimCLRDownstream, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50,}
        

        self.backbone2 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
        dim_mlp2 = self.backbone2.fc.in_features
        
        # If you are using multimodal data you can un-comment the following lines:
        # self.backbone1 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
        # dim_mlp1 = self.backbone1.fc.in_features
        
        # add final linear layer
        self.fc = torch.nn.Linear(dim_mlp2, out_dim, bias=True)
        # self.fc = torch.nn.Linear(dim_mlp1 + dim_mlp2, out_dim, bias=True)

        # self.backbone1.fc = torch.nn.Identity()
        self.backbone2.fc = torch.nn.Identity()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x2 = self.backbone2(x["s2"])

        # If you are using multimodal data you can un-comment the following lines and comment z = self.fc(x2):
        # x1 = self.backbone1(x["s1"])
        # z = torch.cat([x1, x2], dim=1)
        # z = self.fc(z)
     
        z = self.fc(x2)
        
        return z
    
    def load_trained_state_dict(self, weights):
        """load the pre-trained backbone weights"""
        
        # remove the MLP projection heads
        for k in list(weights.keys()):
            if k.startswith(('backbone1.fc', 'backbone2.fc')):
                del weights[k]
        
        log = self.load_state_dict(weights, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
        
        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False


if __name__=="__main__":

    """Testing the model"""
    base_model = "resnet50"
    num_classes = 8

    model = eval('DoubleResNetSimCLRDownstream')(base_model, num_classes)
