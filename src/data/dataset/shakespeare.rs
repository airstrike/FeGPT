use super::{LanguageModelDataset, LanguageModelItem};
use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShakespeareItem {
    pub text: String,
}

#[derive(Debug)]
pub struct ShakespeareDataset {
    dataset: SqliteDataset<ShakespeareItem>,
}

impl ShakespeareDataset {
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<ShakespeareItem> =
            HuggingfaceDatasetLoader::new("Trelis/tiny-shakespeare")
                .dataset(split)
                .unwrap();
        Self { dataset }
    }

    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn validation() -> Self {
        Self::new("validation")
    }

    pub fn test() -> Self {
        Self::new("test")
    }
}

impl Dataset<LanguageModelItem> for ShakespeareDataset {
    fn get(&self, index: usize) -> Option<LanguageModelItem> {
        self.dataset
            .get(index)
            .map(|item| LanguageModelItem::new(item.text))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl LanguageModelDataset for ShakespeareDataset {
    fn vocab_size() -> usize {
        // Shakespeare text has a relatively small vocabulary
        // We can use a much smaller vocab than GPT-2's
        8192
    }

    fn get_raw_text(&self, index: usize) -> Option<String> {
        self.dataset.get(index).map(|item| item.text)
    }
}
