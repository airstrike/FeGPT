#[macro_use]
extern crate derive_new;

use burn::data::dataloader::Dataset;
use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;
use burn::train::metric::Adaptor;
use burn::train::ClassificationOutput;
use perplexity::PerplexityInput;
use std::fs;
use std::sync::Arc;

pub mod cli;
pub mod data;
pub mod model;
pub mod perplexity;
pub mod session;
pub mod utils;

use cli::*;
pub use data::tokenizer;
use data::*;
use model::FeGPTConfig;
use session::{ModelConfig, TrainingMetrics};
use tokenizer::GPTTokenizer;

type Elem = burn::tensor::f16;
type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            dataset,
            d_model,
            d_ff,
            n_layer,
            n_head,
            learning_rate,
            warmup_steps,
            block_size,
            batch_size,
            max_iters,
            dropout,
            compile,
            models_dir,
            ..
        } => {
            let (_, _, metrics) = train(
                &dataset,
                d_model,
                d_ff,
                n_layer,
                n_head,
                learning_rate,
                warmup_steps,
                block_size,
                batch_size,
                max_iters,
                dropout,
                compile,
                &models_dir,
            )?;
            println!("\nTraining completed! Final metrics:");
            println!("{}", "-".repeat(60));
            println!("Training metrics:");
            println!("  Loss:       {:.4}", metrics.final_loss);
            println!("  Perplexity: {:.4}", metrics.final_perplexity);
            println!("\nValidation metrics:");
            println!("  Loss:       {:.4}", metrics.validation_loss);
            println!("  Perplexity: {:.4}", metrics.validation_perplexity);
            println!("\nTraining summary:");
            println!("  Total epochs: {}", metrics.total_epochs);
            println!("  Total steps:  {}", metrics.total_steps);
            println!("{}", "-".repeat(60));
        }

        Commands::Generate {
            prompt,
            num_tokens,
            temperature,
            model,
            models_dir,
        } => {
            // Find model by ID or timestamp
            let config = session::find_model(&models_dir, &model)?;

            // Initialize model using the saved config parameters
            let device = if cfg!(target_os = "macos") {
                burn::tensor::Device::<Backend>::Mps
            } else {
                burn::tensor::Device::<Backend>::Cuda(0)
            };

            let tokenizer = Arc::new(GPTTokenizer::default());
            let vocab_size = tokenizer.vocab_size();
            let pad_token = tokenizer.pad_token();

            let transformer_config = TransformerEncoderConfig::new(
                config.get_d_model(),
                config.get_d_ff(),
                config.get_n_layer(),
                config.get_n_head(),
            )
            .with_norm_first(true)
            .with_dropout(0.1);

            let model_config = FeGPTConfig::new(transformer_config, vocab_size, pad_token, 64);
            let mut model = model_config.init::<Backend>(&device);

            // Load the trained weights
            let model_path = config.model_path(&models_dir);
            println!("Attempting to load model from: {}", model_path.display());

            if !model_path.exists() {
                return Err(format!("Model file not found at {}", model_path.display()).into());
            }

            model = model
                .load_file(model_path, &burn::record::CompactRecorder::new(), &device)
                .expect("Failed to load model");

            let generated = generate(&prompt, &model, &tokenizer, num_tokens, temperature);
            println!("Generated text:\n{}", generated);
        }

        Commands::List { models_dir } => match session::list_trained_models(&models_dir) {
            Ok(entries) => {
                println!("\nTrained models:");
                if entries.is_empty() {
                    println!("No trained models found.");
                } else {
                    // Print header with more fields
                    println!("{:<4} {:<20} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6} {:<8} {:<6} {:<8} {:<12} {:<12}", 
                        "ID", "Timestamp", "Duration", "Layers", "d_model", "d_ff", "Heads", 
                        "Block", "Batch", "MaxIter", "LR", "Warmup", "Drop", "Compile", "Loss", "Perplexity"
                    );
                    println!("{}", "-".repeat(180));

                    for entry in entries {
                        let config = entry.config;
                        println!(
                            "{:<4} {:<20} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6.0e} {:<8} {:<6.3} {:<8} {:<12.4} {:<12.4}",
                            entry.id,
                            config.get_timestamp(),
                            config.get_duration().as_ref().map_or("", |d| d),
                            config.get_n_layer(),
                            config.get_d_model(),
                            config.get_d_ff(),
                            config.get_n_head(),
                            config.get_block_size(),
                            config.get_batch_size(),
                            config.get_max_iters(),
                            config.get_learning_rate(),
                            config.get_warmup_steps(),
                            config.get_dropout(),
                            config.get_compile(),
                            entry.metrics.as_ref().map_or(f64::NAN, |m| m.validation_loss),
                            entry.metrics.as_ref().map_or(f64::NAN, |m| m.validation_perplexity),
                        );
                    }
                    println!("\nNote: Loss and Perplexity shown are validation metrics");
                }
            }
            Err(e) => {
                println!("Error reading models from directory: {}", e);
                return Err(Box::new(e));
            }
        },
    }
    Ok(())
}

fn generate<B: burn::tensor::backend::Backend>(
    prompt: &str,
    model: &crate::model::FeGPT<B>,
    tokenizer: &Arc<impl Tokenizer>,
    max_new_tokens: usize,
    temperature: f64,
) -> String {
    let device = &model.devices()[0];
    let tokens: Vec<i64> = tokenizer
        .encode(prompt, false)
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let len = tokens.len();

    let input =
        burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(tokens, [1, len]), device);

    let output = model.infer(input, max_new_tokens, temperature);
    let generated_tokens: Vec<usize> = output
        .to_data()
        .to_vec()
        .unwrap()
        .into_iter()
        .map(|x: i64| x as usize)
        .collect();

    tokenizer.decode(&generated_tokens)
}

fn train(
    dataset_name: &str,
    d_model: usize,
    d_ff: usize,
    n_layer: usize,
    n_head: usize,
    learning_rate: f64,
    warmup_steps: usize,
    block_size: usize,
    batch_size: usize,
    max_iters: usize,
    dropout: f64,
    compile: bool,
    models_dir: &str,
) -> std::io::Result<(
    crate::model::FeGPT<Backend>,
    Arc<GPTTokenizer>,
    TrainingMetrics,
)> {
    // Create config before training
    let mut config = ModelConfig::new(
        d_model,
        d_ff,
        n_layer,
        n_head,
        learning_rate,
        warmup_steps,
        block_size,
        batch_size,
        max_iters,
        dropout,
        compile,
    );

    let start_time = std::time::Instant::now();

    // Create models directory if it doesn't exist
    fs::create_dir_all(models_dir)?;

    let device = if cfg!(target_os = "macos") {
        burn::tensor::Device::<Backend>::Mps
    } else {
        burn::tensor::Device::<Backend>::Cuda(0)
    };

    let tokenizer = Arc::new(GPTTokenizer::default());
    let vocab_size = tokenizer.vocab_size();
    let pad_token = tokenizer.pad_token();

    let transformer_config = TransformerEncoderConfig::new(d_model, d_ff, n_layer, n_head)
        .with_norm_first(true)
        .with_dropout(dropout);

    let model_config = FeGPTConfig::new(transformer_config, vocab_size, pad_token, block_size);

    let mut optimizer = AdamConfig::new().with_beta_2(0.99).init();

    let mut lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(learning_rate)
        .with_warmup_steps(warmup_steps)
        .with_model_size(d_model)
        .init();

    let dataset_train = DatasetType::new(dataset_name, "train");
    let dataset_test = DatasetType::new(dataset_name, "test");

    // Calculate number of epochs based on max_iters and batch size
    let total_batches = dataset_train.len() / batch_size;
    let epochs = max_iters / total_batches + 1; // +1 to ensure we reach max_iters

    let batcher_train = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), block_size);
    let batcher_test = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), block_size);

    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(8)
        .build(dataset_train);

    let dataloader_test = burn::data::dataloader::DataLoaderBuilder::new(batcher_test)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(8)
        .build(dataset_test);

    let mut model = model_config.init::<Backend>(&device);

    let mut metrics = TrainingMetrics {
        final_loss: f64::NAN,
        final_perplexity: f64::NAN,
        validation_loss: f64::NAN,
        validation_perplexity: f64::NAN,
        total_epochs: 0,
        total_steps: 0,
    };

    let mut step = 0;
    for epoch in 0..epochs {
        // Training loop
        for batch in dataloader_train.iter() {
            let output = model.forward(batch);
            let loss = output.loss.clone();

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            let grads = utils::clip_gradients(grads, &model, 1.0); // Using 1.0 as max_norm like nanoGPT
            model = optimizer.step(learning_rate, model, grads);

            // Update learning rate
            lr_scheduler.step();

            let perplexity_input =
                <ClassificationOutput<Backend> as Adaptor<PerplexityInput>>::adapt(&output);

            // Update metrics
            metrics.final_loss = output.loss.clone().into_scalar().into();
            metrics.final_perplexity = (perplexity_input.loss as f64).exp();
            metrics.total_steps = step;

            step += 1;
            if step >= max_iters {
                break;
            }
        }
        metrics.total_epochs = epoch + 1;

        // Validation loop
        let model_valid = model.valid();
        let mut val_losses = Vec::new();
        let mut val_perplexities = Vec::new();

        for batch in dataloader_test.iter() {
            let output = model_valid.forward(batch);
            val_losses.push(output.loss.clone().into_scalar().into());
            let perplexity_input = <ClassificationOutput<burn::backend::LibTorch<Elem>> as Adaptor<PerplexityInput>>::adapt(&output);
            val_perplexities.push((perplexity_input.loss as f64).exp());
        }

        // Calculate validation metrics
        if !val_losses.is_empty() {
            metrics.validation_loss = val_losses.iter().sum::<f64>() / val_losses.len() as f64;
            metrics.validation_perplexity =
                val_perplexities.iter().sum::<f64>() / val_perplexities.len() as f64;
        }

        println!(
                "Epoch {}/{} - Step {} - Train Loss: {:.4} - Train PPL: {:.4} - Val Loss: {:.4} - Val PPL: {:.4}",
                epoch + 1,
                epochs,
                step,
                metrics.final_loss,
                metrics.final_perplexity,
                metrics.validation_loss,
                metrics.validation_perplexity
            );

        if step >= max_iters {
            break;
        }
    }

    // Update config with training duration and save everything
    config.set_duration(format!("{:.2?}", start_time.elapsed()));
    config.save(models_dir)?;
    config.save_metrics(models_dir, &metrics)?;

    // Save model with timestamp-based name
    model
        .clone()
        .save_file(
            config.model_path(models_dir),
            &burn::record::CompactRecorder::new(),
        )
        .expect("Failed to save model");

    Ok((model, tokenizer, metrics))
}
