import torch
import numpy as np
from torchfm.layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, vocab_size, embed_dim, mlp_dims, dropout):
        super().__init__()
        
        # exclude app_bundle
#        field_dims = np.hstack((field_dims[0:1],field_dims[2:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:])) 
        # include app_bundle
        field_dims = np.hstack((field_dims[:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:])) 
        
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x, additional):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # excldue app_bundle
#        x = torch.cat((x[:,0:1],x[:,2:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1) 
        # include app_bundle
        x = torch.cat((x[:,:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1) 
        
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
