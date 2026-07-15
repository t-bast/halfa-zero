//! A small multi-layer perceptron and training loop.
//!
//! Note that parameters live outside the arena/tape as plain `Tensor`s and are re-added as fresh
//! leaves each step: the tape is disposable scratch space for a single learning step. This isn't
//! the most efficient design, but it's simple and clear, so it's a good starting point until
//! performance becomes an issue.

use crate::autograd::engine::{Tape, Tensor, Value};
use rand::distr::Uniform;
use rand::prelude::*;

/// A fully-connected layer in a neural network: the layer contains `in_dim` neurons and the next
/// layer contains `out_dim` neurons. Each neuron is connected to every neuron of the next layer.
/// This is simply represented by a [in_dim, out_dim] weight matrix and a [1, out_dim] bias.
struct Layer {
    w: Tensor, // [in_dim, out_dim]
    b: Tensor, // [1, out_dim]
    in_dim: usize,
    out_dim: usize,
}

impl Layer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut dyn Rng) -> Self {
        // "Xavier-ish" init: scale by 1/sqrt(in_dim) so activations don't blow up as layers get
        // wider. Small random values break symmetry between neurons (all-equal weights would make
        // every neuron learn the same thing and never differentiate).
        let limit = 1.0 / (in_dim as f64).sqrt();
        // let mut rng = rand::rng();
        let distribution = Uniform::new(-limit, limit).unwrap();
        let weights = (0..in_dim * out_dim)
            .map(|_| distribution.sample(rng))
            .collect();
        let bias = vec![0.0; out_dim]; // biases can safely start at zero
        Layer {
            w: Tensor::from_vec(weights, in_dim, out_dim),
            b: Tensor::from_vec(bias, 1, out_dim),
            in_dim,
            out_dim,
        }
    }
}

/// A multi-layer perceptron, which is a fancy word for just a stack of `Layers`.
/// Every hidden layer uses a non-linear activation function, which is key to help the network
/// learn non-linear patterns: depending on the task, some activations functions will perform
/// better than others (tanh, ReLu, sigmoid, etc).
/// Note that the final layer is left linear (no activation) to ensure that the output can take
/// any real value.
pub struct Mlp<F: Fn(Value) -> Value> {
    layers: Vec<Layer>,
    activation: F,
}

impl<F: Fn(Value) -> Value> Mlp<F> {
    /// The `shape` parameter describes the network shape (each layer's number of neurons).
    /// For example, [1, 16, 16, 1] means:
    /// 1 input -> hidden_layer(16) -> hidden_layer(16) -> 1 output.
    pub fn new(shape: &[usize], activation: F) -> Self {
        let mut rng = rand::rng();
        let mut layers = Vec::new();
        for pair in shape.windows(2) {
            layers.push(Layer::new(pair[0], pair[1], &mut rng));
        }
        Mlp { layers, activation }
    }

    /// Build the forward graph for a single input on a fresh tape, returning the parameter leaf
    /// values (so that we can read their gradients afterwards) and the output value.
    ///
    /// `input` is shape [1, in_dim] (batch = 1). Each layer computes:
    ///     h = activation( x . W + b )      for hidden layers (non-linear activation)
    ///     y = x . W + b                    for the final (linear) layer
    ///
    /// TODO: add support for batching ([batch, in_dim] inputs), which will require bias
    ///     broadcasting (figure out what that means and why it's necessary).
    fn forward(&self, tape: &Tape, input: Tensor) -> (Vec<(Value, Value)>, Value) {
        // We store the (weight, bias) pair of each layer and return them to allow the backward
        // step to read their gradients.
        let mut params = Vec::new();
        let mut x = tape.value(input);

        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let w = tape.value(layer.w.clone());
            let b = tape.value(layer.b.clone());
            let pre = x.mul(&w).add(&b); // x . W + b
            x = if i == last {
                pre
            } else {
                (self.activation)(pre)
            };
            params.push((w, b));
        }
        (params, x)
    }

    /// Perform one full training step on a single (input, target) example.
    ///
    /// Returns the loss value (for logging/debugging).
    /// If the loss explodes or oscillates wildly, the learning rate may be too high.
    /// If the loss improves too slowly, the learning rate may be too low.
    pub fn train_step(&mut self, input_value: f64, target_value: f64, learning_rate: f64) -> f64 {
        let tape = Tape::new();
        let input = Tensor::from_vec(vec![input_value], 1, 1);
        let (params, prediction) = self.forward(&tape, input);

        // We compute the mean squared error (loss function), reduced to a single scalar.
        let target = tape.value(Tensor::from_vec(vec![target_value], 1, 1));
        let diff = prediction.sub(&target);
        let loss = diff.powf(2.0);
        // The loss is a [1, 1] matrix (a scalar), so we can directly apply backward() step on it.
        // This modifies the tape by computing the gradients on each node.
        loss.backward();

        // Gradient-descent step: read each leaf's gradient and nudge the network's weights and biases.
        for (layer, (w_leaf, b_leaf)) in self.layers.iter_mut().zip(params.iter()) {
            layer.w.gradient_descent(&w_leaf.grad(), learning_rate);
            layer.b.gradient_descent(&b_leaf.grad(), learning_rate);
        }

        loss.data_at(0)
    }

    /// Run the network on one input (no training) to inspect results for a 1-input, 1-output
    /// network.
    pub fn predict(&self, x: f64) -> f64 {
        let tape = Tape::new();
        let input = Tensor::from_vec(vec![x], 1, 1);
        let (_, out) = self.forward(&tape, input);
        out.data_at(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn train_sin() {
        // We train a basic neural network to fit the function y=sin(x).
        // Network: 1 -> 16 -> 16 -> 1, using tanh as the activation function.
        let mut net = Mlp::new(&[1, 16, 16, 1], |v| v.tanh());
        let learning_rate = 0.01;

        // Training data: we sample sin over [-pi, pi] and normalize the input by pi so it lands
        // in [-1, 1], which works better with the tanh activation function.
        let n: usize = 64;
        let training_data: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let x = -PI + 2.0 * PI * (i as f64) / (n as f64 - 1.0);
                // The input is normalized in [-1, 1] but the target is the true sin(x)
                (x / PI, x.sin())
            })
            .collect();

        // Repeatedly train over the training data set until loss improves.
        let epochs = 20_000;
        for epoch in 0..epochs {
            // One pass over the dataset, one example at a time (stochastic gradient descent).
            let mut total_loss = 0.0;
            for &(x, y) in &training_data {
                total_loss += net.train_step(x, y, learning_rate);
            }
            if epoch % 1000 == 0 {
                println!(
                    "epoch {:5} avg loss {:.6}",
                    epoch,
                    total_loss / training_data.len() as f64
                );
            }
        }

        // We now look at the result: compare the prediction with the expected sin.
        println!("\n x        pred      true      err");
        let mut rng = rand::rng();
        let distribution = Uniform::new(-1.0, 1.0).unwrap();
        let mut prediction_loss = 0.0;
        for _ in 0..n {
            let x = distribution.sample(&mut rng);
            let y = net.predict(x);
            let expected = (x * PI).sin();
            let err = (y - expected).powi(2);
            println!(
                "{:+.3}   {:+.4}   {:+.4}   {:+.4}",
                x * PI,
                y,
                expected,
                err
            );
            prediction_loss += err;
        }
        assert!((prediction_loss / n as f64) < 0.001);
    }
}
