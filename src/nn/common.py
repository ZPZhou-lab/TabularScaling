import torch
from torch import nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self, 
        dense_size: int,
        sparse_size: int,
        sparse_key_size: int,
        num_hiddens: int,
        dense_activation: str = 'relu',
        col_embedding: bool = True
    ):
        super(InputEmbedding, self).__init__()
        self.dense_size = dense_size
        self.sparse_size = sparse_size
        self.sparse_key_size = sparse_key_size
        self.num_features = dense_size + sparse_size
        self.num_hiddens = num_hiddens
        self.dense_activation = dense_activation
        self.col_embedding = col_embedding
        # build dense activation function
        if dense_activation == 'relu':
            self._dense_act = nn.ReLU()
        elif dense_activation == 'sigmoid':
            self._dense_act = nn.Sigmoid()
        elif dense_activation == 'tanh':
            self._dense_act = nn.Tanh()
        else:
            raise ValueError(f'Unknown dense activation function: {dense_activation}')

        # build dense and sparse embedding
        if dense_size > 0:
            self.dense_embed = nn.Embedding(dense_size, num_hiddens)
        if sparse_size > 0:
            self.sparse_embed = nn.Embedding(sparse_key_size, num_hiddens)
        if col_embedding:
            self.col_embed = nn.Parameter(torch.randn(self.num_features, num_hiddens))
    
    def forward(self, x_dense=None, x_sparse=None):
        """
        x_dense: `(batch_size, dense_size)`
            The dense input features, mapping to dense embedding with shape `(batch_size, dense_size, num_hiddens)`
        x_sparse: `(batch_size, sparse_size)`
            The sparse input features, mapping to sparse embedding with shape `(batch_size, sparse_size, num_hiddens)`
        """
        embed = []
        if self.dense_size > 0 and x_dense is not None:
            shape = (*x_dense.shape, self.num_hiddens)
            # shape (batch, dense_size, num_hiddens)
            dense = torch.broadcast_to(x_dense.unsqueeze(-1), shape) * self.dense_embedd.weight
            embed.append(self._dense_act(dense))
        if self.sparse_size > 0 and x_sparse is not None:
            # shape (batch, num_sparse, num_hiddens)
            embed.append(self.sparse_embed(x_sparse))
        # shape (batch, num_features = dense_size + sparse_size, num_hiddens)
        embed = torch.cat(embed, dim=1)

        if self.col_embedding:
            embed = embed + self.col_embed
        
        return embed
    
class CLSToken(nn.Module):
    def __init__(self, num_hiddens: int):
        super(CLSToken, self).__init__()
        self.num_hiddens = num_hiddens
        self.cls = nn.Parameter(torch.randn(1, 1, num_hiddens))
    
    def forward(self, X):
        return torch.cat([
            torch.broadcast_to(self.cls, (X.shape[0], 1, self.num_hiddens)),
            X
        ], dim=1)

 
class ClassificationHead(nn.Module):
    def __init__(self, num_hiddens: int, num_classes: int):
        super(ClassificationHead, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.proj = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_classes)
        )
    
    def forward(self, X):
        return self.proj(X)