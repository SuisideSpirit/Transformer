import torch
import torch.nn as nn
import sys

from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.model.LayerNormalisation.layerNorm import LayerNormalisation
from src.model.attention.multihead_attention import MultiHeadAttention
from src.model.FeedForward.feedForward import FeedForwardBlock
from src.model.ResidualConnection.residualConnection import Residual_Connection


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        try:
            logger.info("Initializing EncoderBlock")
            self.self_attention_block = self_attention_block
            self.feed_forward_block = feed_forward_block
            self.residual_connections = nn.ModuleList([
                Residual_Connection(self_attention_block.d_model, dropout) for _ in range(2)
            ])
        except Exception as e:
            logger.error("Error occurred during EncoderBlock initialization")
            raise TransformerException(e, sys)

    def forward(self, x, src_mask):
        try:
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
            x = self.residual_connections[1](x, self.feed_forward_block)
            return x
        except Exception as e:
            logger.error("Error occurred during EncoderBlock forward pass")
            raise TransformerException(e, sys)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        try:
            logger.info("Initializing Encoder stack")
            self.layers = layers 
            # We initialize LayerNormalisation with d_model from the first layer
            # Since self.layers has at least 1 layer, we can inspect its block size
            features = layers[0].self_attention_block.d_model if len(layers) > 0 else 512
            self.norm = LayerNormalisation(features)
        except Exception as e:
            logger.error("Error occurred during Encoder initialization")
            raise TransformerException(e, sys)

    def forward(self, x, mask):
        try:
            for layer in self.layers:
                x = layer(x, mask)   
            return self.norm(x)
        except Exception as e:
            logger.error("Error occurred during Encoder forward pass")
            raise TransformerException(e, sys)