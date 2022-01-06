use rand::Rng;
use std::f64::consts::E;
use serde::Serialize;

#[derive(Clone, Serialize)]
pub enum ActivationFunction {
    HyperTan,
    Sigmoidal,
    Step,
    Linear,
    Relu
}

#[derive(Clone, Serialize)]
pub struct Neuron {
    pub output: f64,
    bias: f64,
    pub weights: Vec<f64>,
    activation_function: ActivationFunction
}

impl Neuron {
    pub fn initialize(layer_number: f64, activation_function: ActivationFunction, input_number: u64, use_rand: bool) -> Self{
        let mut weights: Vec<f64> = Vec::new();
        let mut rng = rand::thread_rng();

        if use_rand {
            for _ in 0..input_number - 1 {
                weights.push(rng.gen_range(-10.0, 10.0) as f64);
            }
        } else {
            for _ in 0..input_number - 1 {
                weights.push(0.0);
            }
        }
        Self {
            output: 0f64,
            bias: layer_number - 1f64,
            weights: weights,
            activation_function: activation_function
        }
    }
    pub fn predict(&self, inputs: &Vec<f64>) -> f64 {
        let mut counter = 0;
        let mut input: f64;
        let mut weighted_sum = self.bias;
        for weight in self.weights.iter() {
            input = inputs[counter];
            weighted_sum += weight * input;
            counter += 1;
        }
        self.activate(weighted_sum)
    }
    pub fn adjust(&mut self, actual_q_value: f64, predicted_value: f64, reward: f64, inputs: &Vec<f64>, next_state: &Vec<f64>) {
        let delta = -1.0 * (actual_q_value - predicted_value).powf(2.0);
        for i in 0..self.weights.len() {
            self.weights[i] += delta * (next_state[i]- inputs[i]).powf(2.0) * reward
        }
    }
    fn activate(&self, weighted_sum: f64) -> f64{
        match self.activation_function {
            ActivationFunction::HyperTan => {
                weighted_sum.tanh()
            },
            ActivationFunction::Sigmoidal => {
                1.0/(1.0 + E.powf(-1.0 * weighted_sum))
            },
            ActivationFunction::Step => {
                if weighted_sum > 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            ActivationFunction::Linear => {
                weighted_sum
            },
            ActivationFunction::Relu => {
                if 0.0 > weighted_sum {
                    return 0.0
                } else {
                    return weighted_sum
                }
            }
        }
    }
}