use crate::neuron::{ Neuron, ActivationFunction };
use std::collections::HashMap;

// #[cfg(test)]
// mod tests {
//     use crate::dqn::Network;
//     #[test]
//     fn check_forward_propagate(){
//         let mut network: Network = Network::generate_network(vec![2, 3, 1], 2);
//         let q_value = network.generate_q_value(&vec![1.0, 5.0]);
//         println!("{}", q_value.1);
//     }

//     #[test]
//     fn check_back_propagate(){
//         let mut network = Network::generate_network(vec![2, 3, 1], 2);
//         network.backpropagate(&vec![2.0, 3.0], 5f64, 10f64, &vec![5.0, 2.0]);
//     }
// }

pub struct Network {
    pub neurons: HashMap<String, Vec<Neuron>>,
    pub iterations_passed: u64
}

impl Clone for Network {
    fn clone(&self) -> Network {
        let keys_copy = self.neurons.keys();
        let mut cloned_hashmap = HashMap::new();
        for key in keys_copy {
            let layer = self.neurons.get(key).unwrap();
            cloned_hashmap.insert(key.clone(), layer.clone());
        }
        Network {
            neurons: cloned_hashmap,
            iterations_passed: self.iterations_passed.clone()
        }
    }
}

impl Network {
    pub fn generate_network(structure: Vec<u64>, input_layer_amount: u64) -> Self {
        let mut neurons: HashMap<String, Vec<Neuron>> = HashMap::new();
        let mut counter = 0;
        for layer in structure.iter(){
            let mut neuron_vec: Vec<Neuron> = Vec::new();
            let layer_name = format!("layer {}", counter + 1);
            for _ in 0..*layer as i64 {
                let inputs: u64;
                let neuron: Neuron;
                if counter == 0 {
                    inputs = input_layer_amount;
                } else {
                    inputs = structure[counter - 1]
                }

                if counter + 1 != structure.len() {
                    neuron = Neuron::initialize((counter + 1usize) as f64, ActivationFunction::Relu, inputs, true);
                } else {
                    neuron = Neuron::initialize((counter + 1) as f64, ActivationFunction::Relu, inputs, false);
                }
                neuron_vec.push(neuron)
            }
            counter += 1;
            neurons.insert(String::from(layer_name), neuron_vec);
        }

        Self {
            neurons: neurons,
            iterations_passed: 0
        }
    }

    pub fn generate_q_value(&mut self, input_state: &Vec<f64>) -> (Vec<f64>, f64){
        let mut outputs: Vec<f64> = Vec::new();
        let mut counter = 0;
        let mut q_vec: Vec<f64> = Vec::new();
        let mut max_q_value: f64 = 0.0;
        let ref_map = self.clone().neurons;
        let mut prev_outputs: Vec<f64>;
        for layer in self.neurons.values_mut() {
            prev_outputs = outputs.clone();
            outputs = Vec::new();
            for neuron in layer {
                if counter == 0 {
                    outputs.push(neuron.predict(&input_state))
                } else{
                    outputs.push(neuron.predict(&prev_outputs))
                }
                if counter == ref_map.keys().len() - 1 {
                    let current_prediction = neuron.predict(&prev_outputs);
                    q_vec.push(current_prediction.clone());
                    if current_prediction > max_q_value {
                        max_q_value = current_prediction;
                    }
                }
            }
            counter += 1;
        }
        return (q_vec, max_q_value)
    }

    pub fn backpropagate(&mut self, inputs: &Vec<f64>, actual_q_value: f64, action: usize, reward: f64, next_state: &Vec<f64>) {
        let predicted_q_value = self.generate_q_value(&inputs);
        let layer_amt = self.neurons.keys().len();
        let mut counter = 1;
        let mut current_index = 1;
        for layer in self.neurons.values_mut(){
            current_index = 0;
            for neuron in layer {
                if counter != layer_amt || current_index == action {
                    neuron.adjust(actual_q_value, predicted_q_value.1, reward, &inputs, &next_state);
                }
                current_index += 1;
            }
            counter += 1;
        }
    }
}