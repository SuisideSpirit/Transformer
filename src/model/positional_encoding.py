import torch.nn as nn
import torch 
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.config.transformer_constants import DIMENSION , MAX_SENTENCE_LENGTH
import sys
import math

class Positional_Encoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = DIMENSION 
        self.max_sentence_length = MAX_SENTENCE_LENGTH

        pe = torch.zeros(
            self.max_seq_length,
            self.d_model
        )
        pe = torch.zeros(self.max_seq_length, self.d_model)

        position = torch.arange(
            self.max_seq_length,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                self.d_model,
                2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
    def forward(self, x:torch.tensor):
        try:
            for i in range(x[0]):
                x[0][i] += self.pos_encoder
            
            return x 
        except Exception as e:
            raise TransformerException(e,sys)
