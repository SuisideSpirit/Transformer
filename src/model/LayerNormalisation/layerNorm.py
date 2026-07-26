import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
import sys


class LayerNormalisation(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        try:
            logger.info(f"Initializing LayerNormalisation with features={features}, eps={eps}")
            self.eps = eps
            self.alpha = nn.Parameter(torch.ones(features))
            self.bias = nn.Parameter(torch.zeros(features))
        except Exception as e:
            logger.error("Error occurred during LayerNormalisation initialization")
            raise TransformerException(e, sys)

    def forward(self, x: torch.Tensor):
        try:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias
        except Exception as e:
            logger.error("Error occurred during LayerNormalisation forward pass")
            raise TransformerException(e, sys)