# Transformer from Scratch

A **PyTorch implementation of the Transformer architecture from scratch**, inspired by the paper **“Attention Is All You Need”**. This project focuses on understanding and implementing the core components of a Transformer without relying on high-level Transformer implementations.

## 🚀 Overview

The goal of this project is to build a Transformer architecture step-by-step to understand how modern sequence-to-sequence models work internally.

Instead of using pre-built Transformer modules, the project implements the major components manually, including:

* Tokenization
* Token embeddings
* Positional encoding
* Query, Key, and Value projections
* Scaled Dot-Product Self-Attention
* Multi-Head Attention
* Layer Normalization
* Residual Connections
* Feed-Forward Neural Networks
* Encoder architecture
* Decoder architecture
* Masking mechanisms
* Sequence-to-sequence training

The implementation is designed to be **modular, readable, and easy to extend**.

---

## 🧠 Transformer Architecture

The Transformer follows the encoder-decoder architecture introduced in:

> **Attention Is All You Need**
> Vaswani et al., 2017

The overall flow is:

```text
Input Text
    │
    ▼
Tokenizer
    │
    ▼
Token Embeddings
    │
    ▼
Positional Encoding
    │
    ▼
┌─────────────────────┐
│      Encoder        │
│                     │
│ Multi-Head Attention│
│         ↓           │
│ Add & Norm           │
│         ↓           │
│ Feed Forward         │
│         ↓           │
│ Add & Norm           │
└─────────────────────┘
    │
    │ Encoder Output
    ▼
┌─────────────────────┐
│      Decoder        │
│                     │
│ Masked Self-Attention│
│         ↓           │
│ Add & Norm           │
│         ↓           │
│ Cross Attention      │
│         ↓           │
│ Add & Norm           │
│         ↓           │
│ Feed Forward         │
│         ↓           │
│ Add & Norm           │
└─────────────────────┘
    │
    ▼
Linear Projection
    │
    ▼
Softmax
    │
    ▼
Output Tokens
```

---

## ✨ Key Components

### 1. Token Embedding

Converts token IDs into dense vector representations.

```text
Token ID → Embedding Vector
```

The main model dimension used in the project is:

```text
d_model = 512
```

---

### 2. Positional Encoding

Since Transformers do not inherently understand token order, positional information is added to the token embeddings.

The implementation uses **sinusoidal positional encoding**:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This allows the model to incorporate information about the position of each token in the sequence.

---

### 3. Self-Attention

Self-attention allows every token to interact with other tokens in the sequence.

The input is transformed into:

```text
Query (Q)
Key   (K)
Value (V)
```

Attention is calculated as:

```text
Attention(Q, K, V)
= softmax(QKᵀ / √dₖ)V
```

This is one of the central mechanisms behind the Transformer architecture.

---

### 4. Multi-Head Attention

Instead of performing attention only once, the model divides the representation into multiple attention heads.

Each head can learn different relationships between tokens.

```text
Input
  │
  ├── Head 1 → Attention
  ├── Head 2 → Attention
  ├── ...
  └── Head N → Attention
          │
          ▼
      Concatenate
          │
          ▼
      Linear Layer
```

---

### 5. Layer Normalization

Layer normalization is applied around the attention and feed-forward blocks to stabilize training.

The architecture follows the standard:

```text
Sublayer
   ↓
Residual Connection
   ↓
Layer Normalization
```

---

### 6. Feed-Forward Network

Each Transformer block contains a position-wise feed-forward neural network.

Conceptually:

```text
Linear
  ↓
Activation
  ↓
Linear
```

The feed-forward layer processes each token representation independently after the attention operation.

---

### 7. Residual Connections

Residual connections allow information from previous layers to bypass the current transformation.

```text
Input ───────────────┐
  │                  │
  ▼                  │
Sublayer             │
  │                  │
  └────── Add ◄──────┘
           │
           ▼
      Layer Norm
```

They help with gradient flow and make deeper Transformer architectures easier to train.

---

### 8. Masking

The decoder uses **causal masking** to prevent the model from seeing future tokens during training.

For example:

```text
I     am    learning    Transformers

I      ✓     ✗           ✗
am     ✓     ✓           ✗
learning ✓   ✓           ✓
```

This ensures that token generation remains autoregressive.

---

## 📁 Project Structure

```text
Transformer/
│
├── src/
│   ├── config/
│   │   └── config.py
│   │
│   ├── components/
│   │   ├── embedding.py
│   │   ├── positional_encoding.py
│   │   ├── attention.py
│   │   ├── multi_head_attention.py
│   │   ├── feed_forward.py
│   │   ├── layer_norm.py
│   │   ├── encoder.py
│   │   └── decoder.py
│   │
│   ├── model/
│   │   └── transformer.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── tokenizer/
│   │   └── tokenizer.py
│   │
│   ├── utils/
│   │   └── ...
│   │
│   ├── logger.py
│   └── exception.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── README.md
└── main.py
```

> The exact structure may evolve as additional Transformer components are implemented.

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **NumPy**
* **Custom Tokenizer**
* **Custom Attention Implementation**
* **Custom Positional Encoding**
* **Git & GitHub**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SuisideSpirit/Transformer.git
cd Transformer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the main training or experimentation script:

```bash
python main.py
```

Depending on the current implementation, configuration such as model dimensions, batch size, number of heads, sequence length, and learning rate can be modified from the configuration files.

---

## 🔬 Implementation Details

The project is being developed incrementally to understand each Transformer component individually.

Current implementation focuses on:

* Custom model configuration
* Custom logging
* Custom exception handling
* Tokenization and vocabulary management
* Token embedding
* Sinusoidal positional encoding
* Query, Key, Value projections
* Scaled dot-product attention
* Layer normalization
* Transformer building blocks

### Example Tensor Dimensions

For a typical attention operation:

```text
Batch Size × Sequence Length × Embedding Dimension

2 × 8 × 512
```

The attention mechanism transforms these representations into Query, Key, and Value matrices and computes attention scores between tokens.

---

## 🧪 Debugging & Validation

Individual components are tested using tensor-shape and numerical checks.

Examples include:

```text
Input Shape
    ↓
Q/K/V Shape
    ↓
Attention Shape
    ↓
Output Shape
```

Layer normalization is also checked to ensure that the output has approximately:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

and that numerical issues such as `NaN` values are not introduced.

---

## 📚 Learning Objectives

This project was created primarily as a **deep learning and Transformer architecture learning project**.

Through this implementation, I am exploring:

1. How self-attention works internally
2. Why Query, Key, and Value representations are required
3. How multi-head attention captures different relationships
4. Why positional encoding is necessary
5. How residual connections improve training
6. Why LayerNorm is used
7. How encoder-decoder Transformers perform sequence-to-sequence learning
8. How causal masking enables autoregressive generation
9. How the individual components combine to form modern Transformer architectures

---

## 🔮 Future Improvements

Planned improvements include:

* [ ] Complete Multi-Head Attention
* [ ] Implement encoder stack
* [ ] Implement decoder stack
* [ ] Implement cross-attention
* [ ] Add padding and causal masks
* [ ] Complete end-to-end training pipeline
* [ ] Add text generation
* [ ] Add beam search
* [ ] Improve tokenizer
* [ ] Add unit tests
* [ ] Add training/evaluation metrics
* [ ] Add GPU support
* [ ] Experiment with different model configurations

---

## 📖 Reference

This project is based primarily on the original Transformer paper:

**Attention Is All You Need**
Ashish Vaswani et al.
2017

The paper introduced the Transformer architecture and demonstrated the effectiveness of self-attention for sequence-to-sequence tasks.

---

## 👨‍💻 Author

**Pratik Singh Negi**

Interested in:

* Machine Learning
* Deep Learning
* Generative AI
* Transformers
* NLP
* Data Structures & Algorithms

---

## ⭐ Acknowledgement

This project is built for educational purposes to understand the internal working of Transformer-based models by implementing their core components from scratch.
