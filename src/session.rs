use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    timestamp: String,
    d_model: usize,
    d_ff: usize,
    n_layer: usize,
    n_head: usize,
    learning_rate: f64,
    warmup_steps: usize,
    epochs: usize,
    subset: bool,
    training_duration: Option<String>,
}

impl ModelConfig {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_layer: usize,
        n_head: usize,
        learning_rate: f64,
        warmup_steps: usize,
        epochs: usize,
        subset: bool,
    ) -> Self {
        Self {
            timestamp: Local::now().format("%Y%m%d_%H%M%S").to_string(),
            d_model,
            d_ff,
            n_layer,
            n_head,
            learning_rate,
            warmup_steps,
            epochs,
            subset,
            training_duration: None,
        }
    }

    pub fn model_path(&self, models_dir: &str) -> PathBuf {
        PathBuf::from(models_dir).join(format!("model_{}", self.timestamp))
    }

    pub fn config_path(&self, models_dir: &str) -> PathBuf {
        PathBuf::from(models_dir).join(format!("model_{}.json", self.timestamp))
    }

    pub fn save(&self, models_dir: &str) -> std::io::Result<()> {
        let config_path = self.config_path(models_dir);
        let config_str = serde_json::to_string_pretty(self)?;
        fs::write(config_path, config_str)?;
        Ok(())
    }

    // Getters for private fields
    pub fn get_timestamp(&self) -> &str {
        &self.timestamp
    }

    pub fn get_duration(&self) -> &Option<String> {
        &self.training_duration
    }

    pub fn get_d_model(&self) -> usize {
        self.d_model
    }

    pub fn get_d_ff(&self) -> usize {
        self.d_ff
    }

    pub fn get_n_layer(&self) -> usize {
        self.n_layer
    }

    pub fn get_n_head(&self) -> usize {
        self.n_head
    }

    pub fn get_epochs(&self) -> usize {
        self.epochs
    }

    pub fn get_subset(&self) -> bool {
        self.subset
    }

    pub fn set_duration(&mut self, duration: String) {
        self.training_duration = Some(duration);
    }
}

pub fn list_trained_models(models_dir: &str) -> std::io::Result<Vec<ModelConfig>> {
    let models_path = PathBuf::from(models_dir);
    let mut configs = Vec::<ModelConfig>::new();

    for entry in fs::read_dir(models_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(config) = serde_json::from_str(&content) {
                    configs.push(config);
                }
            }
        }
    }

    // Sort by timestamp, newest first
    configs.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(configs)
}
