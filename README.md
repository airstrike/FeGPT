# FeGPT 🤖

A minimal yet capable Rust implementation of a GPT-style transformer model using [burn](https://github.com/burn-rs/burn).

## Overview

FeGPT implements a small-scale transformer model focused on text generation. Built to explore transformer architectures in Rust, it provides a complete training and inference pipeline with approximately 14.2M parameters.

Key features:
- Complete transformer architecture implementation
- WikiText-2 dataset integration with extensible dataset system
- Efficient training on consumer hardware (Apple Silicon/CUDA)
- Comprehensive experiment tracking and checkpointing
- Temperature-controlled text generation
- Real-time training metrics and progress monitoring

## Project Structure

```
fegpt/
├── src/
│   ├── cli.rs                 # Command-line interface
│   ├── model.rs               # Transformer implementation
│   ├── data/                  # Data processing
│   │   ├── batcher.rs         # Training batch creation
│   │   ├── dataset/          
│   │   │   ├── wikitext.rs    # WikiText-2 implementation
│   │   │   └── utils.rs       # Dataset utilities
│   │   └── tokenizer/        
│   │       ├── character.rs   # Alternative tokenizer
│   │       └── gpt2.rs        # GPT-2 tokenizer
│   ├── perplexity.rs          # Perplexity metrics
│   └── session.rs             # Training management
└── models/                    # Saved model checkpoints
```

## Installation

Add FeGPT to your `Cargo.toml`:

```toml
[dependencies]
fegpt = { git = "https://github.com/airstrike/FeGPT.git" }
```

## Quick Start

Training a model:
```bash
cargo run -- train --d-model 128 --n-layer 4 --n-head 4 --max-iters 10000
```

Generating text:
```bash
cargo run -- generate --prompt "Once upon a time" --num-tokens 100 --model latest
```

Listing trained models:
```bash
cargo run -- list
```

## Architecture Details

- 4 transformer layers with 4 attention heads each
- 128-dimensional embeddings
- 512-dimensional feed-forward networks
- GPT-2 tokenizer (50,257 vocabulary)
- ~14.2M parameters total
  - Embeddings/projection: ~12.9M
  - Transformer layers: ~1.3M
  - Positional embeddings: ~8K

## Training

Default configuration:
```
Batch size: 12
Context length: 64
Learning rate: 1e-4 
Warmup steps: 1000
Training iterations: 10000
```

Typical results:
- Training time: ~19 minutes (10K iterations)
- Final perplexity: ~31
- Hardware: Apple M2 Max (32GB)

## Development

Building:
```bash
cargo build --release
```

Running tests:
```bash
cargo test
```

## Hardware Requirements

- Apple Silicon with Metal support, or
- NVIDIA GPU with CUDA support
- 32GB RAM recommended

## Known Limitations

- Text generation quality needs improvement
- Limited context window (64 tokens)
- Large embedding layer due to full GPT-2 vocabulary

## Future Work

- Vocabulary size optimization
- Context length extension
- Enhanced attention mechanisms
- Dataset flexibility (Tiny Shakespeare support)
- Performance optimization for consumer hardware

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [burn](https://github.com/burn-rs/burn) - Rust ML framework
- [tokenizers](https://github.com/huggingface/tokenizers) - Hugging Face tokenizers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.