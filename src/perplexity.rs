use burn::prelude::Backend;
use burn::train::metric::{Adaptor, Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::ClassificationOutput;

#[derive(Clone, Debug)]
pub struct PerplexityMetric {
    total_loss: f64,
    count: usize,
}

#[derive(Debug)]
pub struct PerplexityInput {
    loss: f64,
}

impl Default for PerplexityMetric {
    fn default() -> Self {
        Self {
            total_loss: 0.0,
            count: 0,
        }
    }
}

impl Metric for PerplexityMetric {
    const NAME: &'static str = "perplexity";
    type Input = PerplexityInput;

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.total_loss += item.loss;
        self.count += 1;

        let perplexity = if self.count == 0 {
            f64::INFINITY
        } else {
            (self.total_loss / self.count as f64).exp()
        };

        MetricEntry::new(
            Self::NAME.to_string(),
            format!("Perplexity: {:.4}", perplexity),
            perplexity.to_string(),
        )
    }

    fn clear(&mut self) {
        self.total_loss = 0.0;
        self.count = 0;
    }
}

// Update the Adaptor implementation to handle f16
impl<B: Backend> Adaptor<PerplexityInput> for ClassificationOutput<B>
where
    B::FloatElem: Into<f64>,
{
    fn adapt(&self) -> PerplexityInput {
        let loss_scalar: f64 = self.loss.clone().into_scalar().into();
        PerplexityInput { loss: loss_scalar }
    }
}

impl Numeric for PerplexityMetric {
    fn value(&self) -> f64 {
        if self.count == 0 {
            f64::INFINITY
        } else {
            (self.total_loss / self.count as f64).exp()
        }
    }
}
