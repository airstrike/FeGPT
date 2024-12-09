use crate::model::FeGPT;
use burn::tensor::backend::AutodiffBackend;

pub fn clip_gradients<B: AutodiffBackend>(
    mut grads: burn::optim::GradientsParams,
    model: &FeGPT<B>,
    max_norm: f32,
) -> burn::optim::GradientsParams
where
    B::FloatElem: Into<f32>,
{
    let mut total_norm: f32 = 0.0;

    let param_ids = burn::module::list_param_ids(model);

    for id in param_ids.iter() {
        if let Some(grad) = grads.get::<B, 1>(id.clone()) {
            let norm: f32 = grad
                .clone()
                .powf_scalar(2.0)
                .sum()
                .sqrt()
                .into_scalar()
                .into();
            total_norm += norm * norm;
        }
    }
    total_norm = total_norm.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for id in param_ids {
            if let Some(grad) = grads.remove::<B, 1>(id.clone()) {
                let scaled_grad = grad * scale;
                grads.register(id, scaled_grad);
            }
        }
    }

    grads
}
