pub mod neuron;
pub mod agent;
pub mod dqn;
pub mod convolution_layer;

// #[cfg(test)]
// mod tests {
//     use crate::StateStorage;
//     #[test]
//     fn get_state_storage(){
//         let state_storage = StateStorage::new(vec![2, 2, 2, 5], 5u64, 4u64, 5, 7);
//         println!("{}", state_storage.get_agent())
//     }
// }

use agent::Agent;
use wasm_bindgen::prelude::*;
use serde_json;

#[wasm_bindgen]
pub struct StateStorage {
    agent: Agent
}


#[wasm_bindgen]
impl StateStorage {
    pub fn new(structure: Vec<u64>, amount_of_states: u64, iteration_backprop: u64, min_value: u8, max_value: u8, epsilon: f64) -> StateStorage {
        StateStorage {
            agent: Agent::initialize_agent(structure, amount_of_states, iteration_backprop, min_value..max_value, epsilon)
        }
    }

    pub fn from_string(json_input: String) -> StateStorage {
        let agent: Agent = serde_json::from_str(json_input.as_str()).unwrap();
        StateStorage {
            agent: agent
        }
    }

    pub fn get_action(&mut self, current_state: Vec<f64>) -> usize {
        self.agent.exploit_explore(&current_state)
    }
    
    pub fn act(&mut self, current_state: Vec<f64>, action: usize, next_state: Vec<f64>, reward: f64){
        self.agent.act(current_state, action, next_state, reward);
    }

    pub fn get_agent(&self) -> String{
        return serde_json::to_string(&self.agent).unwrap();
    }
}

pub trait Policy {
    fn policy(arguments: Vec<f64>) -> f64;
}
