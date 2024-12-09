// dataset/mod.rs
// mod aesop;
mod shakespeare;
mod utils;
mod wikitext;

// pub use aesop::AesopDataset;
pub use shakespeare::ShakespeareDataset;
pub use utils::SubDataset;
pub use wikitext::WikiText2Dataset;

use burn::data::dataset::Dataset;

#[derive(new, Clone, Debug)]
pub struct LanguageModelItem {
    pub text: String,
}

#[allow(dead_code)]
pub trait LanguageModelDataset: Dataset<LanguageModelItem> {
    fn vocab_size() -> usize
    where
        Self: Sized;
    fn get_raw_text(&self, index: usize) -> Option<String>;
}

#[derive(Debug)]
pub enum DatasetType {
    WikiText2(WikiText2Dataset),
    Shakespeare(ShakespeareDataset),
}

impl DatasetType {
    pub fn new(dataset_name: &str, split: &str) -> Self {
        match dataset_name {
            "wikitext" => Self::WikiText2(WikiText2Dataset::new(split)),
            "shakespeare" => Self::Shakespeare(ShakespeareDataset::new(split)),
            _ => panic!("Unknown dataset type: {}", dataset_name),
        }
    }
}

impl Dataset<LanguageModelItem> for DatasetType {
    fn get(&self, index: usize) -> Option<LanguageModelItem> {
        match self {
            DatasetType::WikiText2(dataset) => dataset.get(index),
            DatasetType::Shakespeare(dataset) => dataset.get(index),
        }
    }

    fn len(&self) -> usize {
        match self {
            DatasetType::WikiText2(dataset) => dataset.len(),
            DatasetType::Shakespeare(dataset) => dataset.len(),
        }
    }
}

impl LanguageModelDataset for DatasetType {
    fn vocab_size() -> usize {
        // Could make this dynamic based on the variant,
        // but for now using the largest vocab size to be safe
        50257
    }

    fn get_raw_text(&self, index: usize) -> Option<String> {
        match self {
            DatasetType::WikiText2(dataset) => dataset.get_raw_text(index),
            DatasetType::Shakespeare(dataset) => dataset.get_raw_text(index),
        }
    }
}
