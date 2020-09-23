import torch
import numpy as np
from torchfm.layer import FieldAwareFactorizationMachine, MultiLayerPerceptron, FeaturesLinear


class FieldAwareNeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Neural Factorization Machine.

    Reference:
        L Zhang, et al. Field-aware Neural Factorization Machine for Click-Through Rate Prediction, 2019.
    """

    def __init__(self, field_dims, vocab_size, embed_dim, mlp_dims, dropouts):
        super().__init__()
        
        # except app_bundle
#        field_dims = np.hstack((field_dims[0:1],field_dims[2:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        # include app_bundle
        field_dims = np.hstack((field_dims[:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)
        self.ffm_output_dim = len(field_dims) * (len(field_dims) - 1) // 2 * embed_dim
        self.bn = torch.nn.BatchNorm1d(self.ffm_output_dim)
        self.dropout = torch.nn.Dropout(dropouts[0])
        self.mlp = MultiLayerPerceptron(self.ffm_output_dim, mlp_dims, dropouts[1])

    def forward(self, x, additional):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # except app_bundle
#        x = torch.cat((x[:,0:1],x[:,2:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        # include app_bundle
        x = torch.cat((x[:,:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        
        cross_term = self.ffm(x).view(-1, self.ffm_output_dim)
        cross_term = self.bn(cross_term)
        cross_term = self.dropout(cross_term)
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))
