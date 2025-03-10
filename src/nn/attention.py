import torch
from torch import nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float=0.0,
        latent_dim: int=None,
        bias: bool=False
    ):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        if latent_dim is not None:
            assert latent_dim % num_heads == 0, 'latent_dim must be divisible by num_heads'
            self.latent_dim = latent_dim
        else:
            assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
            self.latent_dim = embed_dim
        self.head_dim = self.latent_dim // self.num_heads
        # build query, key, and value projection
        self.proj_q = nn.Linear(embed_dim, self.latent_dim, bias=bias)
        self.proj_k = nn.Linear(embed_dim, self.latent_dim, bias=bias)
        self.proj_v = nn.Linear(embed_dim, self.latent_dim, bias=bias)
        self.proj_out = nn.Linear(self.latent_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        """
        query:  `(batch_size, tgt_len, embed_dim)`
        key:    `(batch_size, src_len, embed_dim)`
        value:  `(batch_size, src_len, embed_dim)`
        """
        B, tgt_len, src_len = query.size(0), query.size(1), key.size(1)
        # shape (tgt_len | src_len, batch_size, latent_dim)
        q = self.proj_q(query.transpose(0, 1))
        k = self.proj_k(key.transpose(0, 1))
        v = self.proj_v(value.transpose(0, 1))
        q = q.view(tgt_len, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, B * self.num_heads, self.head_dim).transpose(0, 1)

        # scaled dot-product attention
        q_scaled = q * math.sqrt(1.0 / float(self.head_dim))
        # (B * num_heads, tgt_len, src_len)
        attn_weights = torch.bmm(q_scaled, k.transpose(-2, -1)) 
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        attn_output = torch.bmm(attn_weights, v) # (B * num_heads, tgt_len, head_dim)
        # transpose back
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * B, self.latent_dim)
        attn_output = self.proj_out(attn_output)
        attn_output = attn_output.view(tgt_len, B, self.embed_dim)
        attn_output = attn_output.transpose(0, 1).contiguous()

        # process attn_weights
        attn_weights = attn_weights.view(B, self.num_heads, tgt_len, src_len)

        return attn_output, attn_weights

        # # fast implementation without attn weights
        # # (batch, n_head, tokens, latent_dim // num_heads)
        # q = q.view(B, tgt_len, self.num_heads, self.latent_dim // self.num_heads).transpose(1, 2)
        # k = k.view(B, src_len, self.num_heads, self.latent_dim // self.num_heads).transpose(1, 2)
        # v = v.view(B, src_len, self.num_heads, self.latent_dim // self.num_heads).transpose(1, 2)
        # out = F.scaled_dot_product_attention(q, k, v)
        # out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        # out = self.proj_out(out)
    
    
class ScaledDotProductSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float=0.0,
        latent_dim: int=None,
        bias: bool=False
    ):
        super(ScaledDotProductSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        if latent_dim is not None:
            assert latent_dim % num_heads == 0, 'latent_dim must be divisible by num_heads'
            self.latent_dim = latent_dim
        else:
            assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
            self.latent_dim = embed_dim
        self.head_dim = self.latent_dim // self.num_heads
        # build query, key, and value projection
        self.proj_in = nn.Linear(embed_dim, 3 * self.latent_dim, bias=bias)
        self.proj_out = nn.Linear(self.latent_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, X):
        """
        X:  `(batch_size, seq_len, embed_dim)`
        """
        B, seq_len = X.size(0), X.size(1)
        # shape (seq_len, batch_size, latent_dim)
        qkv = self.proj_in(X.transpose(0, 1))
        q, k, v = qkv.chunk(3, dim=-1) # (seq_len, batch_size, latent_dim)
        q = q.contiguous().view(seq_len, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(seq_len, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, B * self.num_heads, self.head_dim).transpose(0, 1)

        # scaled dot-product attention
        q_scaled = q * math.sqrt(1.0 / float(self.head_dim))
        # (B * num_heads, seq_len, seq_len)
        attn_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)
        
        attn_output = torch.bmm(attn_weights, v)
        # transpose back
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len * B, self.latent_dim)
        attn_output = self.proj_out(attn_output)
        attn_output = attn_output.view(seq_len, B, self.embed_dim)
        attn_output = attn_output.transpose(0, 1).contiguous()

        # process attn_weights
        attn_weights = attn_weights.view(B, self.num_heads, seq_len, seq_len)
        
        return attn_output, attn_weights