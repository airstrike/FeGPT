#[macro_use]
extern crate derive_new;

pub mod data;
pub use data::tokenizer;
pub mod model;

use burn::data::dataloader::Dataset;
use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use std::sync::Arc;

use crate::data::{LanguageModelBatcher, SubDataset, Tokenizer, WikiText2Dataset};
use crate::model::FeGPTConfig;
use crate::tokenizer::GPTTokenizer;

type Elem = burn::tensor::f16;

type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

#[allow(dead_code)]
fn full_main() {
    let (trained_model, tokenizer) = run_training(true); // train with subset = true
    let generated = generate("Once upon a time", &trained_model, &tokenizer, 100, 0.8);
    println!("Generated text:\n{}", generated);
}

fn main() {
    // Instead of training, we just load the model
    let device = if cfg!(target_os = "macos") {
        burn::tensor::Device::<Backend>::Mps
    } else {
        burn::tensor::Device::<Backend>::Cuda(0)
    };

    let tokenizer = Arc::new(GPTTokenizer::default());
    let vocab_size = tokenizer.vocab_size();
    let pad_token = tokenizer.pad_token();

    let transformer_config = TransformerEncoderConfig::new(
        128, // d_model (embedding dimension)
        512, // d_ff (feed forward dimension = 4x embedding dim)
        4,   // n_layer
        4,   // n_head
    )
    .with_norm_first(true);

    let config = FeGPTConfig::new(transformer_config, vocab_size, pad_token, 512);

    let artifact_dir = "/tmp/fegpt";

    let mut model = config.init::<Backend>(&device);
    model = model
        .load_file(
            format!("{artifact_dir}/model"),
            &burn::record::CompactRecorder::new(),
            &device,
        )
        .expect("Failed to load model");

    let generated = generate("Once upon a time", &model, &tokenizer, 100, 0.8);
    println!("Generated text:\n{}", generated);
}

fn generate<B: burn::tensor::backend::Backend>(
    prompt: &str,
    model: &crate::model::FeGPT<B>,
    tokenizer: &GPTTokenizer,
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

#[allow(dead_code)]
fn run_training(subset: bool) -> (crate::model::FeGPT<Backend>, Arc<GPTTokenizer>) {
    let device = if cfg!(target_os = "macos") {
        burn::tensor::Device::<Backend>::Mps
    } else {
        burn::tensor::Device::<Backend>::Cuda(0)
    };

    let tokenizer = Arc::new(GPTTokenizer::default());
    let vocab_size = tokenizer.vocab_size();
    let pad_token = tokenizer.pad_token();

    let transformer_config = TransformerEncoderConfig::new(128, 512, 4, 4).with_norm_first(true);

    let config = FeGPTConfig::new(transformer_config, vocab_size, pad_token, 512);

    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1.0e-6)))
        .init();

    let lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(1e-4)
        .with_warmup_steps(2000)
        .with_model_size(128)
        .init();

    let artifact_dir = "/tmp/fegpt";
    std::fs::create_dir_all(artifact_dir).unwrap();

    let dataset_train = WikiText2Dataset::train();
    let dataset_test = WikiText2Dataset::test();
    let dataset_length = dataset_train.len();

    let dataset_train = if subset {
        SubDataset::new(dataset_train, 1_000)
    } else {
        SubDataset::new(dataset_train, dataset_length)
    };

    let batcher_train = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), 512);
    let batcher_test = LanguageModelBatcher::new(tokenizer.clone(), device.clone(), 512);

    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(batcher_train)
        .batch_size(16)
        .shuffle(42)
        .num_workers(12) // Increase workers to match your CPU core count
        .build(dataset_train);

    let dataloader_test = burn::data::dataloader::DataLoaderBuilder::new(batcher_test)
        .batch_size(16)
        .shuffle(42)
        .num_workers(12) // Same here for test loader
        .build(dataset_test);

    let model = config.init::<Backend>(&device);

    let learner = burn::train::LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(burn::train::metric::LossMetric::new())
        .metric_valid_numeric(burn::train::metric::LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(5)
        .build(model, optimizer, lr_scheduler);

    let trained_model = learner.fit(dataloader_train, dataloader_test);

    trained_model
        .clone()
        .save_file(
            format!("{artifact_dir}/model"),
            &burn::record::CompactRecorder::new(),
        )
        .expect("Failed to save model");

    (trained_model, tokenizer)
}
