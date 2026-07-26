import torch.nn as nn
import torch
from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.model.LayerNormalisation.layerNorm import LayerNormalisation
from src.model.attention.multihead_attention import MultiHeadAttention
from src.model.FeedForward.feedForward import FeedForwardBlock
from src.model.ResidualConnection.residualConnection import Residual_Connection


class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int)-> None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size) 

    def forward(self, x) :
        return torch.log_softmax(self.proj(x) , dim = -1 )
