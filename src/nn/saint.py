import torch
from torch import nn
from .common import (
    InputEmbedding,
    CLSToken,
    ClassificationHead
)
from .attention import (
    ScaledDotProductAttention,
    ScaledDotProductSelfAttention
)
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SAINTOutput:
    logits: torch.Tensor
    loss: torch.Tensor

class ColAttentionBlock(nn.Module):
    """
    For inputs `(batch_size, num_cols, num_hiddens)`, calculate column-wise attention (on `axis=1`)
    """
    def __init__(self, 
        num_hiddens: int, 
        num_heads: int=8, 
        latent_dim: int=None,
        bias: bool=False,
        ffn_hiddens_factor: int=1,
        ffn_dropout: float=0.0,
        attn_dropout: float=0.0
    ):
        super(ColAttentionBlock, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.attn = ScaledDotProductSelfAttention(
            num_hiddens, num_heads, latent_dim=latent_dim,
            bias=bias, dropout=attn_dropout)
        self.ln1 = nn.LayerNorm(num_hiddens)
        self.ln2 = nn.LayerNorm(num_hiddens)
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens * ffn_hiddens_factor),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(num_hiddens * ffn_hiddens_factor, num_hiddens)
        )
        self._attn_weights = None
    
    def forward(self, X):
        """
        X: `(batch_size, num_cols, num_hiddens)`
            run MSA to get output with the same shape of X
        """
        # self-attention
        attn, self._attn_weights = self.attn(X)
        X = self.ln1(attn + X) # residual connection
        # feed-forward network
        X = self.ln2(self.ffn(X) + X) # residual connection

        return X


class RowAttentionBlock(nn.Module):
    """
    For inputs `(batch_size, num_cols, num_hiddens)`, calculate row-wise attention (on `axis=0`)
    """
    def __init__(self,
        num_features: int,
        num_hiddens: int,
        num_heads: int=8,
        latent_dim: int=None,
        bias: bool=False,
        cross_cols: bool=True,
        ffn_hiddens_factor: int=1,
        ffn_dropout: float=0.0,
        attn_dropout: float=0.0
    ):
        super(RowAttentionBlock, self).__init__()
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.cross_cols = cross_cols
        # calculate the num_hiddens
        attn_num_hiddens = num_hiddens * num_features if cross_cols else num_hiddens
        self.attn = ScaledDotProductSelfAttention(
            attn_num_hiddens, num_heads, latent_dim=latent_dim,
            bias=bias, dropout=attn_dropout)
        self.ln1 = nn.LayerNorm(num_hiddens)
        self.ln2 = nn.LayerNorm(num_hiddens)
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens * ffn_hiddens_factor),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(num_hiddens * ffn_hiddens_factor, num_hiddens)
        )
        self._attn_weights = None

    def forward(self, X):
        """
        X: `(batch_size, num_cols, num_hiddens)`
            run MSA to get output with the same shape of X
        """
        # self-attention
        if self.cross_cols:
            B, C, H = X.shape
            # calculate attention cross all columns
            X_ = X.view(1, B, C * H)
            attn, self._attn_weights = self.attn(X_)
            self._attn_weights = self._attn_weights[0]
            attn = attn.view(B, C, H)
        else:
            # calculate attention for each column
            X_ = X.transpose(0, 1)
            attn, self._attn_weights = self.attn(X_)
            attn = attn.transpose(0, 1)
        # residual connection
        X = self.ln1(attn + X)
        # feed-forward network
        X = self.ln2(self.ffn(X) + X)

        return X


class SAINTEncoder(nn.Module):
    def __init__(self, 
        num_features: int,
        num_hiddens: int,
        add_cls_token: bool=True,
        num_layers: int=1, 
        num_heads: int=8, 
        inter_sample: bool=True,
        cross_cols: bool=True,
        ffn_hiddens_factor: int=4,
        ffn_dropout: float=0.0,
        attn_dropout: float=0.0,
        col_attn_latent_dim: int=None,
        row_attn_latent_dim: int=None
    ):
        super(SAINTEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.inter_sample = inter_sample
        self.add_cls_token = add_cls_token
        # create MHA layers
        self.col_attns = nn.ModuleList([
            ColAttentionBlock(
                num_hiddens, num_heads, 
                latent_dim=col_attn_latent_dim,
                ffn_hiddens_factor=ffn_hiddens_factor, 
                ffn_dropout=ffn_dropout, attn_dropout=attn_dropout
            ) for _ in range(num_layers)
        ])
        if inter_sample:
            num_features = num_features + 1 if add_cls_token else num_features
            self.row_attns = nn.ModuleList([
                RowAttentionBlock(
                    num_features, num_hiddens, num_heads,
                    latent_dim=row_attn_latent_dim,
                    cross_cols=cross_cols,
                    ffn_hiddens_factor=ffn_hiddens_factor,
                    ffn_dropout=ffn_dropout, attn_dropout=attn_dropout
                ) for _ in range(num_layers)
            ])
        # create CLS token
        if self.add_cls_token:
            self.cls_token = CLSToken(num_hiddens)
    
    def forward(self, inputs):
        """
        X: `(batch_size, num_cols, num_hiddens)`
        """
        X = self.cls_token(inputs) if self.add_cls_token else inputs
        for i in range(self.num_layers):
            X = self.col_attns[i](X)
            if self.inter_sample:
                X = self.row_attns[i](X)

        # return cls token
        if self.add_cls_token:
            cls_hiddens, hiddens = X[:, 0], X[:, 1:]
        else:
            cls_hiddens, hiddens = None, X
        
        return (cls_hiddens, hiddens)


class SAINTClassifier(nn.Module):
    def __init__(self,
        dense_size: int,
        sparse_size: int,
        sparse_key_size: int,
        num_hiddens: int,
        num_classes: int,
        num_layers: int=1,
        num_heads: int=8,
        inter_sample: bool=True,
        cross_cols: bool=True,
        ffn_hiddens_factor: int=4,
        ffn_dropout: float=0.0,
        attn_dropout: float=0.0,
        col_attn_latent_dim: int=None,
        row_attn_latent_dim: int=None,
        col_embedding: bool=True,
        dense_activation: str='relu'
    ):
        super(SAINTClassifier, self).__init__()
        num_features = dense_size + sparse_size
        # create input embedding
        self.input_embedding = InputEmbedding(
            dense_size, sparse_size, sparse_key_size, num_hiddens,
            col_embedding=col_embedding, dense_activation=dense_activation
        )
        self.encoder = SAINTEncoder(
            num_features, num_hiddens, add_cls_token=True,
            num_layers=num_layers, num_heads=num_heads,
            inter_sample=inter_sample, cross_cols=cross_cols,
            ffn_hiddens_factor=ffn_hiddens_factor,
            ffn_dropout=ffn_dropout, attn_dropout=attn_dropout,
            col_attn_latent_dim=col_attn_latent_dim,
            row_attn_latent_dim=row_attn_latent_dim
        )
        self.class_head = ClassificationHead(num_hiddens, num_classes)
    
    def forward(self, inputs, targets=None):
        hiddens = self.input_embedding(*inputs)
        cls_hiddens, hiddens = self.encoder(hiddens)
        logits = self.class_head(cls_hiddens) # (batch_size, num_classes)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return SAINTOutput(logits, loss)
    
    def build_optimizer(self, 
        lr: float=1e-3, 
        weight_decay: float=1e-4
    ):
        params = self.parameters()
        optimizer = torch.optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=weight_decay
        )
        return optimizer