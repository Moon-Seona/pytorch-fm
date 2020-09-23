import torch
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
import numpy as np
import torch.nn.functional as F

class RecurrentAutoencoder(torch.nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim=16):
        super(RecurrentAutoencoder, self).__init__()

        self.char_embedding_dim = 4
        self.hidden_dim = embed_dim

        self.char_embedding = torch.nn.Embedding(vocab_size, self.char_embedding_dim)

        self.encoder = torch.nn.LSTM(self.char_embedding_dim, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.decoder = torch.nn.LSTM(self.char_embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)

        self.linear = torch.nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):

        embed_x = self.char_embedding(x)

        output, (h, c) = self.encoder(embed_x)

        decoder_input = torch.zeros((embed_x.size(0), 1, self.char_embedding_dim)).to('cuda:0')
        decoder_hidden = h
        decoder_context = c

        output2 = torch.zeros((embed_x.size(0), embed_x.size(1), self.hidden_dim)).to('cuda:0')

        for di in range(x.size(1)):
            decoder_output, (decoder_hidden, decoder_context) = self.decoder(decoder_input, (decoder_hidden[-1].unsqueeze(0), decoder_context[-1].unsqueeze(0)))
            #print(decoder_output.shape)
            topv, topi = self.linear(decoder_output).topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_input = self.char_embedding(decoder_input).view(embed_x.size(0), -1, self.char_embedding_dim)
            output2[:,di,:] = decoder_output.squeeze(1)

        #output2, (h2, c2) = self.decoder(torch.zeros(output.size()).to("cuda:0"), (h, c))
        # print(output2.shape, h.shape)

        return self.linear(output2), h[-1]

class DeepFactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, vocab_size, embed_dim, mlp_dims, dropout):
        super().__init__()
        
        self.char_embedding_dim = embed_dim
        self.embed_dim = embed_dim
        self.output_dim = 1

        self.ae_dim = 256 # autoencoder embedding dim
        
        field_dims = np.hstack((field_dims[:3],field_dims[5:8],field_dims[10:15],field_dims[17:19], field_dims[21:24], field_dims[26:])) 

        # for linear
        self.fc = torch.nn.Embedding(sum(field_dims), self.output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((self.output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.rnn1 = torch.nn.LSTM(self.char_embedding_dim, embed_dim, num_layers=4, batch_first=True) # linear

        self.char_embedding = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.char_embedding_dim)
        
        #self.autoencoder = RecurrentAutoencoder(149, vocab_size, self.ae_dim) # appbundle
        self.autoencoder2 = RecurrentAutoencoder(50, vocab_size, self.ae_dim) # carrier

        #self.encoder_linear = torch.nn.Linear(self.ae_dim, embed_dim)
        self.encoder_linear2 = torch.nn.Linear(self.ae_dim, embed_dim)

        #self.encoder_linear3 = torch.nn.Linear(embed_dim, embed_dim)
        #self.encoder_linear4 = torch.nn.Linear(embed_dim, embed_dim)

        self.encoder_linear5 = torch.nn.Linear(embed_dim, embed_dim)
        self.encoder_linear6 = torch.nn.Linear(embed_dim, embed_dim)
        
        #self.encoder_linear7 = torch.nn.Linear(self.ae_dim, embed_dim)
        
        #self.encoder_linear8 = torch.nn.Linear(embed_dim, embed_dim)
        #self.encoder_linear9 = torch.nn.Linear(embed_dim, embed_dim)

        # when use lstm
        self.embed_output_dim = (len(field_dims)+1) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.char_embedding.weight)

    def forward(self, x, additional):
        
        x = torch.cat((x[:,:3],x[:,5:8],x[:,10:15],x[:,17:19],x[:,21:24], x[:,26:]), dim=1) 
        embed_x = self.embedding(x)
        
        pred_seq2, hidden2 = self.autoencoder2(additional[:,149:199])
        encoded2 = self.encoder_linear6(F.relu(self.encoder_linear5(F.relu(self.encoder_linear2(hidden2.view(-1,1, self.ae_dim))))))

        #pred_seq, hidden = self.autoencoder(additional[:,:149])
        #pred_seq2, hidden2 = self.autoencoder2(additional[:,149:199])
        #pred_seq3, hidden3 = self.autoencoder3(additional[:,199:])

        #encoded = self.encoder_linear4(F.relu(self.encoder_linear3(F.relu(self.encoder_linear(hidden.view(-1,1, self.ae_dim))))))
        #encoded2 = self.encoder_linear6(F.relu(self.encoder_linear5(F.relu(self.encoder_linear2(hidden2.view(-1,1, self.ae_dim))))))
        #encoded3 = self.encoder_linear9(F.relu(self.encoder_linear8(F.relu(self.encoder_linear7(hidden3.view(-1,1, self.ae_dim))))))
        
        embed_x = torch.cat([embed_x[:,:4,:], encoded2, embed_x[:,4:,:]], dim=1)

        # for linear
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        linear_x = (torch.sum(torch.sum(embed_x, dim=2), dim=1) + self.bias).unsqueeze(1)
        
        # when no use ifa and ip, modify self.embed_output_dim in __init__
        x = linear_x + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim)) # batch x embed_dim + batch_size x 1
        
        return torch.sigmoid(x.squeeze(1)), pred_seq2, hidden2, encoded2.squeeze(1), embed_x[:,4,:] # pred_seq2, hidden2, pred_seq3, hidden3,
#        return torch.sigmoid(x.squeeze(1))