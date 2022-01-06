mod neuron;
mod agent;
mod dqn;
// mod equation;

use agent::{ Agent };
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
pub struct StateStorage {
    agent: Agent
}


#[wasm_bindgen]
impl StateStorage {
    pub fn new(structure: Vec<u64>, amount_of_states: u64, copy_amount: u64, min_value: u8, max_value: u8) -> StateStorage {
        StateStorage {
            agent: Agent::initialize_agent(structure, amount_of_states, copy_amount, min_value..max_value)
        }
    }

    pub fn get_action(&mut self, current_state: Vec<f64>) -> usize {
        self.agent.exploit_explore(&current_state)
    }
    
    pub fn act(&mut self, current_state: Vec<f64>, action: usize, next_state: Vec<f64>, reward: f64){
        self.agent.act(current_state, action, next_state, reward);
    }
}

pub trait Policy {
    fn policy(arguments: Vec<f64>) -> f64;
}