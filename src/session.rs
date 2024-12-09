use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    timestamp: String,
    d_model: usize, // n_embd in nanoGPT
    d_ff: usize,    // feed-forward dimension
    n_layer: usize,
    n_head: usize,
    learning_rate: f64,
    warmup_steps: usize,
    block_size: usize, // context length
    batch_size: usize,
    max_iters: usize,
    dropout: f64,
    training_duration: Option<String>,
    compile: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub final_loss: f64,
    pub final_perplexity: f64,
    pub validation_loss: f64,
    pub validation_perplexity: f64,
    pub total_epochs: usize,
    pub total_steps: usize,
}

impl ModelConfig {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_layer: usize,
        n_head: usize,
        learning_rate: f64,
        warmup_steps: usize,
        block_size: usize,
        batch_size: usize,
        max_iters: usize,
        dropout: f64,
        compile: bool,
    ) -> Self {
        Self {
            timestamp: Local::now().format("%Y%m%d_%H%M%S").to_string(),
            d_model,
            d_ff,
            n_layer,
            n_head,
            learning_rate,
            warmup_steps,
            block_size,
            batch_size,
            max_iters,
            dropout,
            training_duration: None,
            compile,
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

    pub fn metrics_path(&self, models_dir: &str) -> PathBuf {
        self.model_dir(models_dir).join("metrics.json")
    }

    pub fn save_metrics(&self, models_dir: &str, metrics: &TrainingMetrics) -> std::io::Result<()> {
        let metrics_path = self.metrics_path(models_dir);
        let metrics_str = serde_json::to_string_pretty(metrics)?;
        fs::write(metrics_path, metrics_str)?;
        Ok(())
    }

    pub fn save(&self, models_dir: &str) -> std::io::Result<()> {
        let model_dir = self.model_dir(models_dir);
        fs::create_dir_all(&model_dir)?;

        let config_path = self.config_path(models_dir);
        let config_str = serde_json::to_string_pretty(self)?;
        fs::write(config_path, config_str)?;
        Ok(())
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn get_max_iters(&self) -> usize {
        self.max_iters
    }

    pub fn get_dropout(&self) -> f64 {
        self.dropout
    }

    pub fn get_compile(&self) -> bool {
        self.compile
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

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn get_warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    pub fn set_duration(&mut self, duration: String) {
        self.training_duration = Some(duration);
    }
}

#[derive(Debug)]
pub struct ModelEntry {
    pub id: usize,
    pub config: ModelConfig,
    pub metrics: Option<TrainingMetrics>,
}

pub fn list_trained_models(models_dir: &str) -> std::io::Result<Vec<ModelEntry>> {
    let models_path = PathBuf::from(models_dir);
    let mut entries = Vec::new();

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
            let metrics_path = path.join("metrics.json");

            if config_path.exists() {
                if let Ok(content) = fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str::<ModelConfig>(&content) {
                        let metrics = if metrics_path.exists() {
                            fs::read_to_string(&metrics_path)
                                .ok()
                                .and_then(|content| serde_json::from_str(&content).ok())
                        } else {
                            None
                        };

                        entries.push((config, metrics));
                    }
                }
            }
        }
    }

    // Sort by timestamp, newest first
    entries.sort_by(|(a, _), (b, _)| b.get_timestamp().cmp(a.get_timestamp()));

    // Create ModelEntries with IDs
    Ok(entries
        .into_iter()
        .enumerate()
        .map(|(id, (config, metrics))| ModelEntry {
            id,
            config,
            metrics,
        })
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
