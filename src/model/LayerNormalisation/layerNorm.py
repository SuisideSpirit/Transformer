import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.config.transformer_constants import DIMENSION, MAX_SENTENCE_LENGTH
import sys
import math

def LayerNormalisation():
    def __init__(self, eps : float = 1e8 ):
        super().__init__()
        self.eps = eps 
        self.alpha = nn.parameter(torch.ones[1]) # multiplies 
        self.bias = nn.parameter(torch.zeros[1]) # added 
            
    def forward(self , x : torch.Tensor ):
        try:
            mean =  x.mean(dim = -1 , keepdim = True)
            std = x.std(dim = -1 , keepdim= True )
            return self.alpha * (x-mean) / (std + self.eps) + self.bias
        except Exception as e:
            logger.error(f"Error occurred during Normalisation")
            raise TransformerException(e, sys)