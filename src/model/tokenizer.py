import os
import sys
from typing import List
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from src.utils.exception import TransformerException
from src.utils.logger import logger

class TransformerTokenizer:
    def __init__(self, max_length: int = 512):
        """
        Args:
            max_length: The maximum sequence length. Inputs will be padded or truncated to this size.
        """
        self.max_length = max_length
        self.tokenizer = None
        
        # Core special tokens required for structural signaling in Transformers
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
    def train(self, files: List[str], vocab_size: int = 30000):
        """
        Trains a BPE tokenizer from scratch on a list of raw text files.
        """
        try:
            logger.info(f"Starting tokenizer training on files: {files} with vocab size {vocab_size}")
            
            # 1. Initialize an empty BPE model
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            
            # 2. Split text by whitespace before applying BPE merging rules
            self.tokenizer.pre_tokenizer = Whitespace()
            
            # 3. Define the trainer configuration
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=self.special_tokens
            )
            
            # 4. Train the model on raw text files
            self.tokenizer.train(files, trainer)
            logger.info("Tokenizer training completed successfully.")
            
        except Exception as e:
            logger.error(f"Error occurred while training the tokenizer: {str(e)}")
            raise TransformerException(e, sys)
            
    def save(self, path: str):
        """Saves the trained tokenizer as a single JSON configuration file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.tokenizer.save(path)
            logger.info(f"Tokenizer saved successfully at {path}")
        except Exception as e:
            raise TransformerException(e, sys)
            
    def load(self, path: str):
        """Loads a pre-trained tokenizer JSON file from disk."""
        try:
            self.tokenizer = Tokenizer.from_file(path)
            logger.info(f"Tokenizer loaded successfully from {path}")
        except Exception as e:
            raise TransformerException(e, sys)
            
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a batch of sentences into a single, uniform PyTorch tensor.
        Handles padding and truncation automatically under the hood.
        
        Args:
            texts: A list of raw strings (e.g., ["I love AI", "Deep learning is cool"])
        Returns:
            A 2D PyTorch Integer Tensor of shape (batch_size, max_length)
        """
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer has not been trained or loaded yet.")
                
            # Configure runtime dynamic padding and truncation boundaries
            self.tokenizer.enable_truncation(max_length=self.max_length)
            self.tokenizer.enable_padding(
                length=self.max_length,
                pad_id=self.tokenizer.token_to_id("[PAD]")
            )
            
            # Process text batch through the compiled BPE engine
            encoded_outputs = self.tokenizer.encode_batch(texts)
            
            # Extract the raw list of numerical integer IDs
            batch_ids = [output.ids for output in encoded_outputs]
            
            # Convert into a PyTorch LongTensor ready for the Embedding Layer
            return torch.tensor(batch_ids, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error occurred during batch encoding: {str(e)}")
            raise TransformerException(e, sys)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id("[PAD]") if self.tokenizer else 0