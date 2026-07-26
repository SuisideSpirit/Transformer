import sys
import torch
import torch.nn as nn

from src.utils.exception import TransformerException
from src.utils.logger import logger
from src.model.LayerNormalisation.layerNorm import LayerNormalisation
from src.model.attention.multihead_attention import MultiHeadAttention
from src.model.FeedForward.feedForward import FeedForwardBlock
from src.model.ResidualConnection.residualConnection import Residual_Connection
from src.model.transformers.decoder import Decoder, DecoderBlock 
from src.model.transformers.encoder import Encoder, EncoderBlock
from src.model.encoding.positional_encoding import Positional_Encoding
from src.model.ResidualConnection.projection import ProjectionLayer
from src.model.encoding.embedding import InputEmbeddings


class Transformer(nn.Module): 

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, trg_embed: InputEmbeddings, src_pos: Positional_Encoding, trg_pos: Positional_Encoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        try:
            logger.info("Initializing Transformer Seq2Seq model wrapper")
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_embed
            self.trg_embed = trg_embed
            self.src_pos = src_pos
            self.trg_pos = trg_pos
            self.projection_layer = projection_layer
        except Exception as e:
            logger.error("Error occurred during Transformer initialization")
            raise TransformerException(e, sys)

    def encode(self, src, src_mask):
        try:
            src = self.src_embed(src)
            src = self.src_pos(src)
            return self.encoder(src, src_mask)
        except Exception as e:
            logger.error("Error occurred during Transformer.encode")
            raise TransformerException(e, sys)
    
    def decode(self, encoder_output, src_mask, trg, trg_mask):
        try:
            trg = self.trg_embed(trg)
            trg = self.trg_pos(trg)
            return self.decoder(trg, encoder_output, src_mask, trg_mask)
        except Exception as e:
            logger.error("Error occurred during Transformer.decode")
            raise TransformerException(e, sys)

    def project(self, decoder_output):
        try:
            return self.projection_layer(decoder_output)
        except Exception as e:
            logger.error("Error occurred during Transformer.project")
            raise TransformerException(e, sys)
    

def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> Transformer:
    try:
        logger.info(f"Building Transformer: src_vocab={src_vocab_size}, trg_vocab={trg_vocab_size}, src_seq_len={src_seq_len}, tgt_seq_len={tgt_seq_len}")
        
        # InputEmbeddings constructor expects: (vocab, dimension)
        src_embed = InputEmbeddings(src_vocab_size, d_model)
        trg_embed = InputEmbeddings(trg_vocab_size, d_model)

        src_pos = Positional_Encoding(d_model, src_seq_len, dropout)
        trg_pos = Positional_Encoding(d_model, tgt_seq_len, dropout)

        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
            encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
            encoder_blocks.append(block)

        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
            decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
            decoder_blocks.append(block)

        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        projection_layer = ProjectionLayer(d_model, trg_vocab_size)

        transformer_model = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)

        for p in transformer_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        return transformer_model
        
    except Exception as e:
        logger.error("Error occurred while building Transformer")
        raise TransformerException(e, sys)
