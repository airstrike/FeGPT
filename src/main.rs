#[macro_use]
extern crate derive_new;

use burn::data::dataloader::Dataset;
use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use std::fs;
use std::sync::Arc;

pub mod data;
pub use data::tokenizer;
pub mod cli;
pub mod model;
pub mod session;

use cli::*;
use data::*;
use model::FeGPTConfig;
use session::ModelConfig;
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
            let (_, _) = train(
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
            println!("Training completed!");
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
                    println!(
                        "{:<4} {:<20} {:<10} {:<8} {:<8} {:<8} {:<8} {:<6} {:<8} {:<6} {:<8}",
                        "ID",
                        "Timestamp",
                        "Duration",
                        "Layers",
                        "d_model",
                        "Heads",
                        "Block",
                        "Batch",
                        "MaxIter",
                        "Drop",
                        "Compile",
                    );
                    println!("{}", "-".repeat(130));

                    for entry in entries {
                        let config = entry.config;
                        println!(
                            "{:<4} {:<20} {:<10} {:<8} {:<8} {:<8} {:<8} {:<6} {:<8} {:<6.3} {:<8}",
                            entry.id,
                            config.get_timestamp(),
                            config.get_duration().as_ref().map_or("", |d| d),
                            config.get_n_layer(),
                            config.get_d_model(),
                            config.get_n_head(),
                            config.get_block_size(),
                            config.get_batch_size(),
                            config.get_max_iters(),
                            config.get_dropout(),
                            config.get_compile(),
                        );
                    }
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
) -> std::io::Result<(crate::model::FeGPT<Backend>, Arc<GPTTokenizer>)> {
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

    let optimizer = AdamConfig::new().with_beta_2(0.99).init();

    let lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(learning_rate)
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

    let model = model_config.init::<Backend>(&device);

    let learner = burn::train::LearnerBuilder::new(models_dir)
        .metric_train_numeric(burn::train::metric::LossMetric::new())
        .metric_valid_numeric(burn::train::metric::LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, optimizer, lr_scheduler);

    let trained_model = learner.fit(dataloader_train, dataloader_test);

    // Update config with training duration and save
    config.set_duration(format!("{:.2?}", start_time.elapsed()));
    config.save(models_dir)?;

    // Save model with timestamp-based name
    trained_model
        .clone()
        .save_file(
            config.model_path(models_dir),
            &burn::record::CompactRecorder::new(),
        )
        .expect("Failed to save model");

    Ok((trained_model, tokenizer))
}
