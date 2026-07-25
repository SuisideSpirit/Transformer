from src.utils.exception import TransformerException
from src.utils.logger import logger
import torch
import torch.nn as nn 
import sys 
import math

class Self_Attention(nn.Module):
    def __init__(self , d_model):
        logger.info("Initilised Self attention")
        super().__init__() 
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self , x : torch.Tensor):
        try:
            logger.info("started the attention mechanism")
            logger.info(f"Input Shape : {x.shape}")
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)

            scores = torch.matmul(Q,K.transpose(-2, -1))

            scores = scores / math.sqrt(self.d_model) ## weights 

            attention = torch.softmax(scores,dim=-1)

            output = torch.matmul(attention,V)

            return output

        except Exception as e :
            raise TransformerException(e,sys)
