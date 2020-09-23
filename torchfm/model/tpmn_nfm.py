import torch
import numpy as np 
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear


class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, vocab_size, embed_dim, mlp_dims, dropouts):
        super().__init__()
        
        # except app_bundle
#        field_dims = np.hstack((field_dims[0:1],field_dims[2:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        # include app_bundle
        field_dims = np.hstack((field_dims[:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x, additional):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
#        x = torch.cat((x[:,0:1],x[:,2:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        # include app_bundle
        x = torch.cat((x[:,:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        
        cross_term = self.fm(self.embedding(x)) # bi-interaction pooling??
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))
