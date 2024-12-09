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

#[derive(new)]
pub struct SubDataset<D> {
    dataset: D,
    limit: usize,
}

impl<D: burn::data::dataset::Dataset<LanguageModelItem>>
    burn::data::dataset::Dataset<LanguageModelItem> for SubDataset<D>
{
    fn get(&self, index: usize) -> Option<LanguageModelItem> {
        if index < self.limit {
            self.dataset.get(index)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        std::cmp::min(self.dataset.len(), self.limit)
    }
}
