import torch
import numpy as np
from torchfm.layer import FeaturesEmbedding, FeaturesLinear, InnerProductNetwork, \
    OuterProductNetwork, MultiLayerPerceptron


class ProductNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.
    Reference:
        Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.
    """

    def __init__(self, field_dims,vacab_size, embed_dim, mlp_dims, dropout, method='inner'):
        super().__init__()
        
        # except app_bundle
        field_dims = np.hstack((field_dims[0:1],field_dims[2:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        # include app_bundle
#        field_dims = np.hstack((field_dims[:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        
        num_fields = len(field_dims)
        if method == 'inner':
            self.pn = InnerProductNetwork()
        elif method == 'outer':
            self.pn = OuterProductNetwork(num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = num_fields * embed_dim
        self.mlp = MultiLayerPerceptron(num_fields * (num_fields - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x, additional):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # except app_bundle
        x = torch.cat((x[:,0:1],x[:,2:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        # include app_bundle
#        x = torch.cat((x[:,:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        
        embed_x = self.embedding(x)
        cross_term = self.pn(embed_x)
        x = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(x)
        return torch.sigmoid(x.squeeze(1))
