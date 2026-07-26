import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
import sys


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        try:
            logger.info(f"Initializing FeedForwardBlock with d_model={d_model}, d_ff={d_ff}, dropout={dropout}")
            self.linear_1 = nn.Linear(d_model, d_ff)
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.Linear(d_ff, d_model)
        except Exception as e:
            logger.error("Error occurred during FeedForwardBlock initialization")
            raise TransformerException(e, sys)

    def forward(self, x: torch.Tensor):
        try:
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        except Exception as e:
            logger.error("Error occurred during FeedForwardBlock forward pass")
            raise TransformerException(e, sys)