use super::LanguageModelItem;
use burn::data::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct SubDataset<D> {
    dataset: D,
    limit: usize,
    indices: Vec<usize>,
}

impl<D> SubDataset<D> {
    pub fn new_with_seed(dataset: D, limit: usize, seed: Option<u64>) -> Self
    where
        D: Dataset<LanguageModelItem>,
    {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        let limit = std::cmp::min(limit, dataset.len());

        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        indices.shuffle(&mut rng);
        indices.truncate(limit);
        indices.sort_unstable();

        Self {
            dataset,
            limit,
            indices,
        }
    }

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
