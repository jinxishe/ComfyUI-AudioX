import torch
from torch import nn
from typing import Callable, Tuple

class Residual(nn.Module):
    """Simple residual connection class"""
    def __init__(self, num_streams: int = 1, dim: int = None):
        super().__init__()
        self.num_streams = num_streams
        
    def forward(self, x):
        if self.num_streams == 1:
            return x, lambda tokens: tokens
        else:
            return x, lambda tokens: tokens

class HyperConnections(nn.Module):
    """Hyper connection class for handling multi-stream residual connections"""
    def __init__(self, num_streams: int, dim: int):
        super().__init__()
        self.num_streams = num_streams
        self.dim = dim
        
        if num_streams > 1:
            self.expand = nn.Linear(dim, dim * num_streams)
            self.reduce = nn.Linear(dim * num_streams, dim)
    
    def forward(self, x):
        if self.num_streams == 1:
            return x, lambda tokens: tokens
        
        expanded = self.expand(x)
        expanded = expanded.view(*x.shape[:-1], self.num_streams, self.dim)
        
        return expanded, lambda tokens: self.reduce(tokens.view(*tokens.shape[:-2], -1))
    
    @staticmethod
    def get_expand_reduce_stream_functions(num_streams: int, disable: bool = False):
        """Get functions for expanding and reducing streams"""
        if disable or num_streams == 1:
            return lambda x: x, lambda x: x
        
        def expand_streams(x):
            if x.dim() == 3:  # [batch, seq, dim]
                return x.unsqueeze(2).expand(-1, -1, num_streams, -1)
            elif x.dim() == 4:  # [batch, seq, streams, dim]
                return x
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        def reduce_streams(x):
            if x.dim() == 4:  # [batch, seq, streams, dim]
                return x.mean(dim=2)
            elif x.dim() == 3:  # [batch, seq, dim]
                return x
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return expand_streams, reduce_streams












