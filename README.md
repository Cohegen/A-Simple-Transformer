# Simple Transformer Model for Text Generation

This repository contains a simple implementation of a Transformer model for text generation using PyTorch. The model is trained on a subset of the WikiText-2 dataset and can generate text based on a given prompt.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Code Structure](#code-structure)
- [License](#license)

## Overview

The Transformer model implemented here is a simplified version of the original Transformer architecture described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The model consists of the following components:

1. **Positional Encoding**: Adds positional information to the input embeddings.
2. **Self-Attention**: Computes attention scores between all positions in the input sequence.
3. **Transformer Block**: Combines self-attention with a feed-forward neural network.
4. **Simple Transformer Model**: Stacks multiple transformer blocks to form the complete model.

The model is trained on a small subset of the WikiText-2 dataset and can generate text based on a given prompt.

## Installation

To run this code, you need to have Python 3.7+ installed along with the following libraries:

- `torch`
- `datasets`
- `tqdm`

You can install the required libraries using pip:

```bash
pip install torch datasets tqdm
```

## Usage

### Training

To train the model, simply run the provided script. The training loop is already set up to train the model for 5 epochs on a small subset of the WikiText-2 dataset.

```python
# Training loop
epochs = 5
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)  # (batch_size, seq_length, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
```

### Inference

After training, you can generate text using the `generate_text` function. The function takes a starting prompt and generates a sequence of tokens based on the trained model.

```python
# Generate and print text
start_tokens = simple_tokenizer("The meaning of life is")
generated = generate_text(model, start_tokens, length=100)
# Convert tokens back to characters
generated_text = ''.join([chr(t) for t in generated if 0 <= t < 128])
print("Generated text:")
print(generated_text)
```

## Code Structure

The code is structured as follows:

- **PositionalEncoding**: Implements positional encoding to add positional information to the input embeddings.
- **SelfAttention**: Implements the self-attention mechanism.
- **TransformerBlock**: Combines self-attention with a feed-forward neural network.
- **SimpleTransformer**: Stacks multiple transformer blocks to form the complete model.
- **WikiTextDataset**: A custom dataset class for loading and processing the WikiText-2 dataset.
- **Training Loop**: The main training loop that trains the model on the dataset.
- **Inference**: Functions to generate text using the trained model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. Happy coding! 🚀
