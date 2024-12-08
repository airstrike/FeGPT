use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    gelu: Gelu,
    dropout: Dropout,
}

impl<B: Backend> MLP<B> {
    pub fn new(n_embd: usize, dropout: f64, device: &B::Device) -> Self {
        // Use 4x expansion ratio as per the original transformer paper
        let d_hidden = n_embd * 4;

        Self {
            fc1: LinearConfig::new(n_embd, d_hidden).init(device),
            fc2: LinearConfig::new(d_hidden, n_embd).init(device),
            gelu: Gelu::new(),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}
