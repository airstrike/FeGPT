use crate::data::TrainingBatch;
use burn::{
    nn::{
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    prelude::*,
    tensor::activation::softmax,
    tensor::backend::AutodiffBackend,
    tensor::TensorData,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use rand::Rng;

#[derive(Config)]
pub struct FeGPTConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl FeGPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeGPT<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        FeGPT {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_length: self.max_seq_length,
        }
    }
}

#[derive(Module, Debug)]
pub struct FeGPT<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl<B: Backend> FeGPT<B> {
    pub fn forward(&self, item: TrainingBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flatten.device());
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }

    pub fn infer(
        &self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Tensor<B, 2, Int> {
        let mut current_seq = idx;
        let device = &self.devices()[0];

        for _ in 0..max_new_tokens {
            // Get sequence context within block_size
            let seq_len = current_seq.dims()[1];
            let idx_cond = if seq_len > self.max_seq_length {
                current_seq.clone().slice([
                    0..current_seq.dims()[0],
                    (seq_len - self.max_seq_length)..seq_len,
                ])
            } else {
                current_seq.clone()
            };

            // Get logits for next token
            let [batch_size, seq_len] = idx_cond.dims();

            // Calculate embeddings
            let index_positions = Tensor::arange(0..seq_len as i64, device)
                .reshape([1, seq_len])
                .repeat_dim(0, batch_size);
            let embedding_positions = self.embedding_pos.forward(index_positions);
            let embedding_tokens = self.embedding_token.forward(idx_cond);
            let embedding = (embedding_positions + embedding_tokens) / 2;

            // Forward pass through transformer
            let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_len, device);
            let encoded = self
                .transformer
                .forward(TransformerEncoderInput::new(embedding).mask_attn(mask_attn));

            // Get logits and apply temperature
            let logits = self.output.forward(encoded);
            let last_logits = logits
                .slice([0..batch_size, (seq_len - 1)..seq_len, 0..self.vocab_size])
                .reshape([batch_size, self.vocab_size]);

            let scaled_logits = if temperature != 1.0 {
                last_logits / temperature
            } else {
                last_logits
            };

            // Convert to probabilities
            let probs = softmax(scaled_logits, 1);

            // Sample next token
            // let next_token = probs.multinomial(1, true);
            let next_token = self.sample_multinomial(probs);

            // Append new token to sequence
            current_seq = Tensor::cat(vec![current_seq, next_token], 1);
        }

        current_seq
    }

    fn sample_multinomial(&self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        let device = probs.device();
        let [batch_size, vocab_size] = probs.dims();

        // Cast the f16 tensor to f32 before extracting the data
        let raw_probs = probs.to_data().to_vec::<burn::tensor::f16>().unwrap();
        let probs: Vec<f32> = raw_probs.iter().map(|x| x.to_f32()).collect();

        // The rest of the code remains the same
        let mut rng = rand::thread_rng();
        let mut next_tokens = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut cumsum = 0.0;
            let rand_val: f32 = rng.gen();
            let mut selected = (vocab_size - 1) as i64;
            let batch_start = batch_idx * vocab_size;
            for token_idx in 0..vocab_size {
                cumsum += probs[batch_start + token_idx];
                if rand_val < cumsum {
                    selected = token_idx as i64;
                    break;
                }
            }

            next_tokens.push(selected.elem::<B::IntElem>());
        }

        Tensor::<B, 1, Int>::from_data(TensorData::from(next_tokens.as_slice()), &device)
            .reshape([batch_size, 1])
    }
}

impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, ClassificationOutput<B>> for FeGPT<B> {
    fn step(&self, item: TrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(item);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainingBatch<B>, ClassificationOutput<B>> for FeGPT<B> {
    fn step(&self, item: TrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}
