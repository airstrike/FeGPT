use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

// Define our base item type for the language model
#[derive(new, Clone, Debug)]
pub struct LanguageModelItem {
    pub text: String,
}

// The raw item structure from WikiText-2 dataset
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WikiText2Item {
    pub text: String,
}

// Main dataset struct holding the SQLite dataset
pub struct WikiText2Dataset {
    dataset: SqliteDataset<WikiText2Item>,
}

// Implement Dataset trait for WikiText2Dataset
impl Dataset<LanguageModelItem> for WikiText2Dataset {
    fn get(&self, index: usize) -> Option<LanguageModelItem> {
        self.dataset
            .get(index)
            .map(|item| LanguageModelItem::new(item.text))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// Implementation of constructors and helpers
impl WikiText2Dataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn validation() -> Self {
        Self::new("validation")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<WikiText2Item> =
            // HuggingfaceDatasetLoader::new("Salesforce/wikitext")
            // 
            // For some reason the Salesforce variant fails to load because no auth?
            // I don't quite follow why, so let's just use this other variant
            HuggingfaceDatasetLoader::new("carlosejimenez/wikitext__wikitext-2-raw-v1")
                .dataset(split)
                .unwrap();
        Self { dataset }
    }
}

// TODO: Implement a trait for language model datasets if we want to add
// dataset-specific functionality? TBD.
#[allow(dead_code)]
pub trait LanguageModelDataset: Dataset<LanguageModelItem> {
    fn vocab_size() -> usize;
    fn get_raw_text(&self, index: usize) -> Option<String>;
}

impl LanguageModelDataset for WikiText2Dataset {
    fn vocab_size() -> usize {
        // WikiText-2 typically uses GPT-2's vocab size
        50257
    }

    fn get_raw_text(&self, index: usize) -> Option<String> {
        self.dataset.get(index).map(|item| item.text)
    }
}

use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct SubDataset<D> {
    dataset: D,
    limit: usize,
    indices: Vec<usize>, // Store our sampled indices
}

impl<D> SubDataset<D> {
    pub fn new_with_seed(dataset: D, limit: usize, seed: Option<u64>) -> Self
    where
        D: Dataset<LanguageModelItem>,
    {
        // Create range of all possible indices
        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        // If limit is larger than dataset, use whole dataset
        let limit = std::cmp::min(limit, dataset.len());

        // Shuffle with provided seed or random seed
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        indices.shuffle(&mut rng);

        // Take only the first 'limit' indices
        indices.truncate(limit);

        // Sort indices for more efficient access patterns
        indices.sort_unstable();

        Self {
            dataset,
            limit,
            indices,
        }
    }

    // Resample the indices (useful for epoch transitions if desired)
    pub fn resample(&mut self, seed: Option<u64>)
    where
        D: Dataset<LanguageModelItem>,
    {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);
        indices.truncate(self.limit);
        indices.sort_unstable();
        self.indices = indices;
    }
}

impl<D: Dataset<LanguageModelItem>> Dataset<LanguageModelItem> for SubDataset<D> {
    fn get(&self, index: usize) -> Option<LanguageModelItem> {
        if index >= self.limit {
            return None;
        }
        self.dataset.get(self.indices[index])
    }

    fn len(&self) -> usize {
        self.limit
    }
}
