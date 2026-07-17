from src.model.embedding import StaticEmbedding
import torch.nn as nn 
import torch 


if __name__ == "__main__":
    embedding_model = StaticEmbedding(6,4) 
    t = torch.tensor([
            [2,3] ,
            [4,1]
        ])
    encoded = embedding_model.forward(t)
    print(encoded[0][1])
