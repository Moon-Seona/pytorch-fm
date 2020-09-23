import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
import numpy as np

import torch.nn.functional as F

class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, vocab_size, embed_dim, mlp_dims, dropout):
        super().__init__()

        # insert code
        #self.char_embedding_dim = embed_dim
        self.embed_dim = embed_dim
        self.output_dim = 1

        self.ae_dim = 256 # autoencoder embedding dim
        
#        field_dims = np.hstack((field_dims[0:1],field_dims[2:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
#        field_dims = np.hstack((field_dims[:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        # exclude carrier
#        field_dims = np.hstack((field_dims[:3],field_dims[5:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
        # include carrier
        field_dims = np.hstack((field_dims[:3],field_dims[4:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:]))
                
        # for linear
        self.fc = torch.nn.Embedding(sum(field_dims), self.output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((self.output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # when use lstm
        self.embed_output_dim = (len(field_dims)) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        #self.init_weights()

    #def init_weights(self):
    #    torch.nn.init.xavier_uniform(self.char_embedding.weight)

    def forward(self, x, additional):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
#        x = torch.cat((x[:,0:1],x[:,2:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1) 
        x = torch.cat((x[:,:3],x[:,4:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        # exclude carrier
#        x = torch.cat((x[:,:3],x[:,5:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1)
        embed_x = self.embedding(x)

        # for linear
        linear_x = (torch.sum(torch.sum(embed_x, dim=2), dim=1) + self.bias).unsqueeze(1)

        x = linear_x + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim)) # batch x embed_dim + batch_size x 1
        
        return torch.sigmoid(x.squeeze(1))