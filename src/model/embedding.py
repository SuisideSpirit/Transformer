import torch.nn as nn
import torch 
from src.utils.exception import TransformerException
from src.utils.logger import logger
import sys


class StaticEmbedding(nn.Module):
    def __init__(self , vocab , dimension):
        super().__init__()
        self.dim = dimension 
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab , dimension)  # making a matrix of vocab * dimension

    def forward(self, x : torch.Tensor):
        try :
            return self.embedding(x) 
        except Exception as e :
            raise TransformerException(e,sys)
