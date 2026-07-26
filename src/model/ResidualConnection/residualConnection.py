import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.model.LayerNormalisation.layerNorm import LayerNormalisation
import sys


class Residual_Connection(nn.Module):
    def __init__(self, features: int, dropout: float = 0.1) -> None:
        super().__init__()
        try:
            logger.info(f"Initializing Residual_Connection with features={features}, dropout={dropout}")
            self.norm = LayerNormalisation(features)
            self.dropout = nn.Dropout(dropout)
        except Exception as e:
            logger.error("Error occurred during Residual_Connection initialization")
            raise TransformerException(e, sys)

    def forward(self, x: torch.Tensor, sublayer):
        try:
            return x + self.dropout(sublayer(self.norm(x))) 
        except Exception as e:
            logger.error("Error occurred during Residual_Connection forward pass")
            raise TransformerException(e, sys)   