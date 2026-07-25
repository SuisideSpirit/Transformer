import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.config.transformer_constants import DIMENSION, DIMENSION_FF
import sys
import math

def FeedForwardBlock():
    def __init__(self,d_model : int = DIMENSION , d_ff : int = DIMENSION_FF , dropout : float = 0.1 ):
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff , d_model) 
            
    def forward(self , x : torch.Tensor ):
        try:
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        except Exception as e:
            logger.error(f"Error occurred during Feed Forward")
            raise TransformerException(e, sys)