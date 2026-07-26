import torch
import torch.nn as nn
import sys

from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.model.LayerNormalisation.layerNorm import LayerNormalisation
from src.model.attention.multihead_attention import MultiHeadAttention
from src.model.FeedForward.feedForward import FeedForwardBlock
from src.model.ResidualConnection.residualConnection import Residual_Connection


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        try:
            logger.info("Initializing DecoderBlock")
            self.self_attention_block = self_attention_block
            self.feed_forward_block = feed_forward_block
            self.cross_attention_block = cross_attention_block
            # The DecoderBlock has 3 sub-layers (Self-Attn, Cross-Attn, FFN), so it needs 3 residual connections.
            self.residual_connections = nn.ModuleList([
                Residual_Connection(self_attention_block.d_model, dropout) for _ in range(3)
            ])
        except Exception as e:
            logger.error("Error occurred during DecoderBlock initialization")
            raise TransformerException(e, sys)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        try:
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x
        except Exception as e:
            logger.error("Error occurred during DecoderBlock forward pass")
            raise TransformerException(e, sys)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        try:
            logger.info("Initializing Decoder stack")
            self.layers = layers 
            features = layers[0].self_attention_block.d_model if len(layers) > 0 else 512
            self.norm = LayerNormalisation(features)
        except Exception as e:
            logger.error("Error occurred during Decoder initialization")
            raise TransformerException(e, sys)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        try:
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)   
            return self.norm(x)
        except Exception as e:
            logger.error("Error occurred during Decoder forward pass")
            raise TransformerException(e, sys)