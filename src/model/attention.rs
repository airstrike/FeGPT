use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    n_head: usize,
    n_embd: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    proj: Linear<B>,
    dropout: Dropout,
    head_size: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(n_embd: usize, n_head: usize, dropout: f64, device: &B::Device) -> Self {
        let head_size = n_embd / n_head;
        assert!(
            head_size * n_head == n_embd,
            "n_embd must be divisible by n_head"
        );

        let query = LinearConfig::new(n_embd, n_embd).init(device);
        let key = LinearConfig::new(n_embd, n_embd).init(device);
        let value = LinearConfig::new(n_embd, n_embd).init(device);
        let proj = LinearConfig::new(n_embd, n_embd).init(device);
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            n_head,
            n_embd,
            query,
            key,
            value,
            proj,
            dropout,
            head_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        // Linear projections and reshape for multi-head
        let q = self
            .query
            .forward(x.clone())
            .reshape([batch_size, seq_len, self.n_head, self.head_size])
            .transpose(1, 2); // [B, H, T, D/H]

        let k = self
            .key
            .forward(x.clone())
            .reshape([batch_size, seq_len, self.n_head, self.head_size])
            .transpose(1, 2); // [B, H, T, D/H]

        let v = self
            .value
            .forward(x)
            .reshape([batch_size, seq_len, self.n_head, self.head_size])
            .transpose(1, 2); // [B, H, T, D/H]

        // Attention scores
        let scores = q.matmul(k.transpose(-2, -1)) // [B, H, T, T]
            / (self.head_size as f64).sqrt();

        // Create causal mask
        let mask = Tensor::tril(seq_len, seq_len, x.device()).reshape([1, 1, seq_len, seq_len]); // [1, 1, T, T]

        let scores = scores.masked_fill(mask.eq(0), f64::NEG_INFINITY);
        let attn = scores.softmax(-1); // [B, H, T, T]
        let attn = self.dropout.forward(attn);

        // Apply attention to values
        let out = attn
            .matmul(v) // [B, H, T, D/H]
            .transpose(1, 2) // [B, T, H, D/H]
            .reshape([batch_size, seq_len, self.n_embd]); // [B, T, D]

        self.proj.forward(out)
    }
}
