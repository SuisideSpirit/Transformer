import torch
import torch.nn as nn
from src.utils.exception import TransformerException
from src.utils.logger import logger
import sys , math 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model : int , h : int , dropout : float):
        super().__init__()
        self.d_model = d_model 
        self.h = h 
        assert d_model%h == 0 , "D_model not divisible by h "

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model , d_model) 
        self.w_k = nn.Linear(d_model , d_model) 
        self.w_v = nn.Linear(d_model , d_model) 

        self.w_o = nn.Linear(d_model , d_model) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query , key , value ,mask ,dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None : 
            attention_scores.masked_fill_(mask == 0 , -1e9)

        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None :
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores


    def forward(self, q , k ,v, mask ):
        try:
            query = self.w_q(q)
            key = self.w_k(k) 
            value = self.w_v(v)

            # (batch , seq_len , d_model) - > (batch , seq_len , h , d_k) -> (batch ,h , seq_len , d_k)
            query = query.view(query.shape[0] , query.shape[1] , self.h , self.d_k).transpose(1,2)
            key = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k).transpose(1,2)
            value= value.view(value.shape[0] , value.shape[1] , self.h , self.d_k).transpose(1,2)

            x , self.attention_scores = MultiHeadAttention.attention(query , key ,value , mask , self.dropout)

            # (batch , seq_len , d_model) <---- (batch , seq_len , h , d_k) <--- (batch ,h , seq_len , d_k)
            x = x.transpose(1,2).contiguous().view(x.shape[0] , x.shape[1] , self.h * self.d_k)

            return self.w_o(x)

        except Exception as e : 
            raise TransformerException(e,sys)