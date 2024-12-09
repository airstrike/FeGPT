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
            subset,
            d_model,
            d_ff,
            n_layer,
            n_head,
            learning_rate,
            warmup_steps,
            epochs,
            models_dir,
            ..  // Handle any additional fields we haven't mapped yet
        } => {
            let (_, _) = train(
                &dataset,
                subset,
                d_model,
                d_ff,
                n_layer,
                n_head,
                learning_rate,
                warmup_steps,
                epochs,
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
                .load_file(
                    model_path,
                    &burn::record::CompactRecorder::new(),
                    &device,
                )
                .expect("Failed to load model");
        
            let generated = generate(&prompt, &model, &tokenizer, num_tokens, temperature);
            println!("Generated text:\n{}", generated);
        }
        
        Commands::List { models_dir } => {
            // Try to create directory if it doesn't exist
            if let Err(e) = fs::create_dir_all(&models_dir) {
                println!("Cannot create directory at `{}`", models_dir);
                return Err(Box::new(e));
            }
        
            match session::list_trained_models(&models_dir) {
                Ok(entries) => {
                    println!("\nTrained models:");
                    if entries.is_empty() {
                        println!("No trained models found.");
                    } else {
                        println!(
                            "{:<4} {:<20} {:<15} {:<10} {:<10} {:<8} {:<8} {:<15}",
                            "ID", "Timestamp", "Duration", "Layers", "d_model", "Heads", "Epochs", "Subset"
                        );
                        println!("{}", "-".repeat(90));
        
                        for entry in entries {
                            let config = entry.config;
                            println!(
                                "{:<4} {:<20} {:<15} {:<10} {:<10} {:<8} {:<8} {:<15}",
                                entry.id,
                                config.get_timestamp(),
                                config.get_duration().as_ref().unwrap_or(&"".to_string()),
                                config.get_n_layer(),
                                config.get_d_model(),
                                config.get_n_head(),
                                config.get_epochs(),
                                config.get_subset()
                            );
                        }
                        println!("\nUse either ID or timestamp when generating text");
                    }
                }
                Err(e) => {
                    println!("Error reading models from directory: {}", e);
                    return Err(Box::new(e));
                }
            }
        }
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
    subset: bool,
    d_model: usize,
    d_ff: usize,
    n_layer: usize,
    n_head: usize,
    learning_rate: f64,
    warmup_steps: usize,
    epochs: usize,
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
        epochs,
        subset,
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
        .with_dropout(0.1);

    let model_config = FeGPTConfig::new(transformer_config, vocab_size, pad_token, 64);

    let optimizer = AdamConfig::new()
        .with_beta_2(0.99)
        // .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1.0e-6)))
        .init();

    let lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(learning_rate)
        .with_warmup_steps(warmup_steps)
        .with_model_size(128)
        .init();

    let dataset_train = DatasetType::new(dataset_name, "train");
    let dataset_test = DatasetType::new(dataset_name, "test");
    let dataset_length = dataset_train.len();

    let dataset_train = if subset {
        SubDataset::new_with_seed(dataset_train, 50_000, Some(42))
    } else {
        SubDataset::new_with_seed(dataset_train, dataset_length, None)
    };

    let batcher_train = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), 64);
    let batcher_test = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), 64);

    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(batcher_train)
        .batch_size(16)
        .shuffle(42)
        .num_workers(8)
        .build(dataset_train);

    let dataloader_test = burn::data::dataloader::DataLoaderBuilder::new(batcher_test)
        .batch_size(16)
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