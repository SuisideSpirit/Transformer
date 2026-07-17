import torch
import torch.nn as nn

from model.attention.self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=2):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One SelfAttention module per head
        self.heads = nn.ModuleList(
            [SelfAttention(self.head_dim) for _ in range(num_heads)]
        )

        # Final projection
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x : (batch_size, seq_len, d_model)
        """

        batch_size, seq_len, _ = x.shape

        # Split embedding dimension into heads
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        head_outputs = []

        for i, head in enumerate(self.heads):
            # Select one head
            head_input = x[:, :, i, :]  # (batch_size, seq_len, head_dim)

            # Run self-attention
            out = head(head_input)

            head_outputs.append(out)

        # Concatenate outputs from all heads
        concat = torch.cat(head_outputs, dim=-1)

        # Final linear projection
        output = self.fc_out(concat)

        return output