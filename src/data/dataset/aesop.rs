// dataset/aesop.rs
use super::{LanguageModelDataset, LanguageModelItem};
use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

const AESOP_FABLES_PATH: &str = "data/aesop";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AesopFable {
    pub title: String,
    pub text: String,
    pub moral: String,
}

#[derive(Debug)]
pub struct AesopDataset {
    fables: Vec<AesopFable>,
    split: String,
}

impl AesopDataset {
    pub fn new(split: &str) -> Self {
        let path = PathBuf::from(AESOP_FABLES_PATH).join(format!("{}.json", split));

        let content = fs::read_to_string(path).expect("Failed to read Aesop's fables dataset");

        let fables: Vec<AesopFable> =
            serde_json::from_str(&content).expect("Failed to parse Aesop's fables JSON");

        Self {
            fables,
            split: split.to_string(),
        }
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

impl Dataset<LanguageModelItem> for AesopDataset {
    fn get(&self, index: usize) -> Option<LanguageModelItem> {
        self.fables.get(index).map(|fable| {
            let full_text = format!(
                "Title: {}\n\n{}\n\nMoral: {}\n",
                fable.title, fable.text, fable.moral
            );
            LanguageModelItem::new(full_text)
        })
    }

    fn len(&self) -> usize {
        self.fables.len()
    }
}

impl LanguageModelDataset for AesopDataset {
    fn vocab_size() -> usize {
        // We can use a smaller vocab size since Aesop's fables have limited vocabulary
        16384
    }

    fn get_raw_text(&self, index: usize) -> Option<String> {
        self.get(index).map(|item| item.text)
    }
}
