import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.model.LayerNormalisation.layerNorm import LayerNormalisation
import sys
import math

class Residual_Connection():
    def __init__(self, dropout : float = 0.1) -> None:
        super().__init__()
        self.norm = LayerNormalisation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor ,sublayer):
        try:
            x = x + self.dropout(sublayer(self.norm(x))) 
        except Exception as e:
            raise TransformerException(e, sys)   