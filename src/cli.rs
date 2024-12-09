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
        #[arg(short, long, default_value = "aesop")]
        dataset: String,

        /// Whether to use a subset of the data for quick testing
        #[arg(short, long)]
        subset: bool,

        /// Model embedding dimension
        #[arg(long, default_value_t = 192)]
        d_model: usize,

        /// Model feed-forward dimension
        #[arg(long, default_value_t = 768)]
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

        /// Number of epochs
        #[arg(short, long, default_value_t = 5)]
        epochs: usize,

        /// Output directory for model artifacts
        #[arg(short, long, default_value = "/tmp/fegpt")]
        output_dir: String,

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
