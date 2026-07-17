import torch
from src.model.tokenizer import TransformerTokenizer
from src.model.embedding import StaticEmbedding  # Using your verified module name
from src.model.positional_encoding import Positional_Encoding

if __name__ == "__main__":

    raw_data_path = "dummy_dataset.txt"
    tokenizer = TransformerTokenizer(max_length=8)
    tokenizer.train([raw_data_path], vocab_size=1000)
    input_text = [
        "I love deep learning.",                              # Short sentence
        "Implementing transformers from scratch is fun."       # Long sentence
    ]
    token_tensors = tokenizer.encode_batch(input_text)
    d_model = 512
    embedding_layer = StaticEmbedding(vocab=tokenizer.vocab_size, dimension=d_model)
    
    dense_embeddings = embedding_layer(token_tensors)
    print(dense_embeddings[1])
    positional_encoding = Positional_Encoding()
    final_vector = positional_encoding.forward(dense_embeddings)
    print(final_vector[1])