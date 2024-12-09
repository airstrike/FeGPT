use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    pub fn model_dir(&self, models_dir: &str) -> PathBuf {
        PathBuf::from(models_dir).join(format!("model_{}", self.timestamp))
    }

    pub fn model_path(&self, models_dir: &str) -> PathBuf {
        self.model_dir(models_dir).join("model.mpk")
    }

    pub fn config_path(&self, models_dir: &str) -> PathBuf {
        self.model_dir(models_dir).join("model.json")
    }

    pub fn save(&self, models_dir: &str) -> std::io::Result<()> {
        let model_dir = self.model_dir(models_dir);
        fs::create_dir_all(&model_dir)?;

        let config_path = self.config_path(models_dir);
        let config_str = serde_json::to_string_pretty(self)?;
        fs::write(config_path, config_str)?;
        Ok(())
    }

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

#[derive(Debug)]
pub struct ModelEntry {
    pub id: usize,
    pub config: ModelConfig,
}

pub fn list_trained_models(models_dir: &str) -> std::io::Result<Vec<ModelEntry>> {
    let models_path = PathBuf::from(models_dir);
    let mut configs = Vec::<ModelConfig>::new();

    for entry in fs::read_dir(models_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.starts_with("model_"))
        {
            let config_path = path.join("model.json");
            if config_path.exists() {
                if let Ok(content) = fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str(&content) {
                        configs.push(config);
                    }
                }
            }
        }
    }

    // Sort by timestamp, newest first
    configs.sort_by(|a, b| b.get_timestamp().cmp(a.get_timestamp()));

    // Create ModelEntries with IDs
    Ok(configs
        .into_iter()
        .enumerate()
        .map(|(id, config)| ModelEntry { id, config })
        .collect())
}

pub fn find_model(models_dir: &str, model_ref: &str) -> std::io::Result<ModelConfig> {
    let models = list_trained_models(models_dir)?;

    // First try to parse as model ID
    if let Ok(id) = model_ref.parse::<usize>() {
        if let Some(entry) = models.iter().find(|e| e.id == id) {
            return Ok(entry.config.clone());
        }
    }

    // Then try as timestamp
    if let Some(entry) = models
        .iter()
        .find(|e| e.config.get_timestamp() == model_ref)
    {
        return Ok(entry.config.clone());
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("No model found with ID or timestamp: {}", model_ref),
    ))
}
