use super::{dataset::LanguageModelItem, tokenizer::Tokenizer};
use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, prelude::*};
use std::sync::Arc;

#[derive(Clone, new)]
pub struct LanguageModelBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    block_size: usize,
}

#[derive(Debug, Clone, new)]
pub struct LanguageModelTrainingBatch<B: Backend> {
    pub tokens_inputs: Tensor<B, 2, Int>, // Input token IDs [batch_size, seq_len]
    pub targets: Tensor<B, 2, Int>,       // Target token IDs [batch_size, seq_len]
    pub mask_pad: Tensor<B, 2, Bool>,     // Attention mask [batch_size, seq_len]
}

impl<B: Backend> Batcher<LanguageModelItem, LanguageModelTrainingBatch<B>>
    for LanguageModelBatcher<B>
{
    fn batch(&self, items: Vec<LanguageModelItem>) -> LanguageModelTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Tokenize all texts
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text, true));
        }

        // Generate padding mask and get padded tokens
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.block_size),
            &self.device,
        );

        let [batch_size, seq_length] = mask.tensor.dims();

        // Create input and target sequences (shifted by 1)
        let input_ids = mask
            .tensor
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let target_ids = mask.tensor.slice([0..batch_size, 1..seq_length]);
        let mask_pad = mask.mask.slice([0..batch_size, 0..seq_length - 1]);

        LanguageModelTrainingBatch::new(
            input_ids.to_device(&self.device),
            target_ids.to_device(&self.device),
            mask_pad.to_device(&self.device),
        )
    }
}

// For inference, we can use string input directly
impl<B: Backend> Batcher<String, LanguageModelTrainingBatch<B>> for LanguageModelBatcher<B> {
    fn batch(&self, items: Vec<String>) -> LanguageModelTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item, true));
        }

        // Generate padding mask
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.block_size),
            &self.device,
        );

        let [batch_size, seq_length] = mask.tensor.dims();

        // Create input and target sequences
        let input_ids = mask
            .tensor
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let target_ids = mask.tensor.slice([0..batch_size, 1..seq_length]);
        let mask_pad = mask.mask.slice([0..batch_size, 0..seq_length - 1]);

        LanguageModelTrainingBatch::new(
            input_ids.to_device(&self.device),
            target_ids.to_device(&self.device),
            mask_pad.to_device(&self.device),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_training_batch_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let tokenizer = Arc::new(super::super::tokenizer::GPTTokenizer::default());
        let batcher = LanguageModelBatcher::<TestBackend>::new(tokenizer, device, 32);

        let items = vec![
            LanguageModelItem::new("Hello world".into()),
            LanguageModelItem::new("Testing batch".into()),
        ];

        let batch = batcher.batch(items);
        let [batch_size, seq_len] = batch.tokens_inputs.dims();
        assert_eq!(batch_size, 2);
        assert!(seq_len < 32);
    }
}
