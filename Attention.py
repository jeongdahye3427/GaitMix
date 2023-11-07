import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class self_attention(nn.Module):
    """
    Self Attention between each of RNN's output and final output
    """
    def __init__(self, size_in, size_out, fact):
        super().__init__()

        inter_med = int(size_in/fact)
        self.attn = nn.Linear(size_in, inter_med)
        self.concat_linear = nn.Linear(2*inter_med, size_out)

    def forward(self, encoder_states, final_hidden_state):
        final_state = self.attn(final_hidden_state)
        attn_weights = torch.bmm(encoder_states, final_state.unsqueeze(2))
        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)
        context = torch.bmm(encoder_states.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        attn_hidden = F.relu(self.concat_linear(torch.cat((context, final_state), dim=1)))
        return attn_hidden

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = HIDDEN_DIM, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        batch_size = value.size(0)
        
#         print(batch_size, self.num_heads, self.d_head)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn
