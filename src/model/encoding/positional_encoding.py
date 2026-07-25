import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.config.transformer_constants import DIMENSION, MAX_SENTENCE_LENGTH
import sys
import math

class Positional_Encoding(nn.Module):
    def __init__(self , d_model :int = DIMENSION, seq_len :int = MAX_SENTENCE_LENGTH, dropout : float = 0.1):
        super().__init__()
        self.dim = DIMENSION
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(self.seq_len, self.dim)

        # create a vector of len(seq _len , 1)
        position = torch.arange(self.seq_len,dtype=torch.float32).unsqueeze(1)

        # denominator of the formula
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32)* (-math.log(10000.0) / self.dim))

        #applying the sine to even position and cosine to odd  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_seq_length, dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        try:
            x = x + (self.pe[: , : x.shape[1] , :]).requires_grad(False)
            return self.dropout(x) 
        except Exception as e:
            raise TransformerException(e, sys)