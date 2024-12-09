pub use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train a new model
    Train {
        /// Dataset to use for training
        #[arg(short, long, default_value = "shakespeare")]
        dataset: String,

        /// Model embedding dimension (n_embd in nanoGPT)
        #[arg(long, default_value_t = 128)]
        d_model: usize,

        /// Model feed-forward dimension
        #[arg(long, default_value_t = 512)]
        d_ff: usize,

        /// Number of transformer layers
        #[arg(long, default_value_t = 4)]
        n_layer: usize,

        /// Number of attention heads
        #[arg(long, default_value_t = 4)]
        n_head: usize,

        /// Learning rate
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,

        /// Number of warmup steps
        #[arg(long, default_value_t = 100)]
        warmup_steps: usize,

        /// Context length (block_size in nanoGPT)
        #[arg(long, default_value_t = 64)]
        block_size: usize,

        /// Batch size for training
        #[arg(long, default_value_t = 12)]
        batch_size: usize,

        /// Maximum number of iterations
        #[arg(long, default_value_t = 2000)]
        max_iters: usize,

        /// Dropout rate
        #[arg(long, default_value_t = 0.0)]
        dropout: f64,

        /// Whether to use torch.compile()
        #[arg(long, default_value_t = false)]
        compile: bool,

        /// Directory to store models
        #[arg(long, default_value = "models")]
        models_dir: String,
    },

    /// Generate text using a trained model
    Generate {
        /// Prompt to start generation with
        #[arg(short, long, default_value = "Once upon a time")]
        prompt: String,

        /// Number of tokens to generate
        #[arg(short, long, default_value_t = 100)]
        num_tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value_t = 0.8)]
        temperature: f64,

        /// Model timestamp to use (e.g., "20240308_123456")
        #[arg(short, long)]
        model: String,

        /// Directory containing models
        #[arg(long, default_value = "models")]
        models_dir: String,
    },
    /// List all trained models
    List {
        /// Directory containing models
        #[arg(long, default_value = "models")]
        models_dir: String,
    },
}
