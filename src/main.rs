#[macro_use]
extern crate derive_new;

pub mod data;
pub use data::tokenizer;
pub mod model;
pub mod sample;
pub mod train;

use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{decay::WeightDecayConfig, AdamConfig};
use burn::prelude::*;
use std::sync::Arc;

use crate::data::{LanguageModelBatcher, Tokenizer, WikiText2Dataset};
use crate::model::FeGPTConfig;
use crate::tokenizer::GPTTokenizer;

type Elem = burn::tensor::f16;

type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() {
    let device = if cfg!(target_os = "macos") {
        // For Apple Silicon support, use Metal Performance Shaders
        burn::tensor::Device::<Backend>::Mps
    } else {
        burn::tensor::Device::<Backend>::Cuda(0)
    };

    // Initialize tokenizer and get necessary values
    let tokenizer = Arc::new(GPTTokenizer::default());
    let vocab_size = tokenizer.vocab_size();
    let pad_token = tokenizer.pad_token();

    // Create transformer config
    let transformer_config = TransformerEncoderConfig::new(
        128, // d_model (embedding dimension)
        512, // d_ff (feed forward dimension = 4x embedding dim)
        4,   // n_layer
        4,   // n_head
    )
    .with_norm_first(true);

    // Model configuration
    let config = FeGPTConfig::new(
        transformer_config,
        vocab_size,
        pad_token,
        512, // max sequence length
    );

    // Training configuration
    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1.0e-6)))
        .init();

    // Learning rate scheduler
    let lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(1e-4)
        .with_warmup_steps(2000)
        .with_model_size(128) // should match d_model from transformer config
        .init();

    // Initialize training artifacts directory
    let artifact_dir = "/tmp/fegpt";
    std::fs::create_dir_all(artifact_dir).unwrap();

    // Initialize datasets
    let dataset_train = WikiText2Dataset::train();
    let dataset_test = WikiText2Dataset::test();

    // Initialize batcher
    let batcher_train = LanguageModelBatcher::new(
        tokenizer.clone(),
        device.clone(),
        512, // max_seq_length
    );
    let batcher_test = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), 512);

    // Initialize dataloaders
    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(batcher_train)
        .batch_size(32)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_test = burn::data::dataloader::DataLoaderBuilder::new(batcher_test)
        .batch_size(32)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_test);

    // Initialize model
    let model = config.init::<Backend>(&device);

    // Initialize learner
    let learner = burn::train::LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(burn::train::metric::LossMetric::new())
        .metric_valid_numeric(burn::train::metric::LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(10)
        .build(model, optimizer, lr_scheduler);

    // Train the model
    let trained_model = learner.fit(dataloader_train, dataloader_test);

    // Save the trained model
    trained_model
        .clone()
        .save_file(
            format!("{artifact_dir}/model"),
            &burn::record::CompactRecorder::new(),
        )
        .expect("Failed to save model");

    // Test generation
    let prompt = "Once upon a time";
    generate_sample(
        &trained_model,
        &tokenizer,
        prompt,
        100, // max_new_tokens
        0.8, // temperature
    );
}

fn generate_sample<B: burn::tensor::backend::Backend>(
    model: &crate::model::FeGPT<B>,
    tokenizer: &GPTTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
) {
    let device = &model.devices()[0];

    // Encode prompt
    let tokens: Vec<i64> = tokenizer
        .encode(prompt, false)
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let len = tokens.len();

    let input =
        burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(tokens, [1, len]), device);

    // Generate
    let output = model.infer(input, max_new_tokens, temperature);

    // Decode and print
    let generated_tokens: Vec<usize> = output
        .to_data()
        .to_vec()
        .unwrap()
        .into_iter()
        .map(|x: i64| x as usize)
        .collect();

    let generated_text = tokenizer.decode(&generated_tokens);
    println!("Generated text:\n{}", generated_text);
}
