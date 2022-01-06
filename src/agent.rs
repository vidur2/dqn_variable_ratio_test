use crate::dqn::Network;
use rand::Rng;
use std::ops::Range;
use serde::{Serialize};

#[derive(Serialize)]
pub struct Agent {
    target_network: Network,
    main_network: Network,
    replay_buffer: Vec<BufferItem>,
    unexplored_actions: Vec<u64>,
    variable_lr: u8,
    variable_range: Range<u8>,
    iteration_backprop: u64
}

#[derive(Serialize)]
struct BufferItem {
    current_state: Vec<f64>,
    next_state: Vec<f64>,
    action: usize,
    reward: f64
}

impl Agent {
    pub fn initialize_agent(structure: Vec<u64>, amount_of_states: u64, iteration_backprop: u64, range: Range<u8>) -> Self {
        let mut rng = rand::thread_rng();
        let main_network = Network::generate_network(structure.clone(), amount_of_states);
        let target_network = main_network.clone();
        let amount_of_actions = structure[structure.len() - 1];
        let mut unexplored_actions: Vec<u64> = Vec::new();
        for i in 1..amount_of_actions + 1 {
            unexplored_actions.push(i);
        }
        Self {
            target_network: target_network,
            main_network: main_network,
            replay_buffer: Vec::new(),
            unexplored_actions: unexplored_actions,
            variable_lr: rng.gen_range(range.clone().start, range.clone().end),
            variable_range: range,
            iteration_backprop: iteration_backprop
        }
    }

    pub fn act(&mut self, current_state: Vec<f64>, action: usize, next_state: Vec<f64>, reward: f64){
        let range = self.variable_range.clone();
        let item = BufferItem {
            current_state: current_state,
            next_state: next_state, 
            action: action,
            reward: reward
        };
        self.replay_buffer.push(item);
        if self.replay_buffer.len() == self.variable_lr as usize {
            for item in self.replay_buffer.iter() {
                let actual_q_value = self.target_network.generate_q_value(&item.current_state);
                self.main_network.backpropagate(&item.current_state, actual_q_value.1, item.action, item.reward, &item.next_state)
            }
            self.main_network.iterations_passed += 1;
            if self.main_network.iterations_passed == self.iteration_backprop {
                let mut rng = rand::thread_rng();
                self.target_network.neurons = self.main_network.neurons.clone();
                self.main_network.iterations_passed = 0;
                self.variable_lr = rng.gen_range(range.start, range.end)
            }
        }
    }

    pub fn exploit_explore(&mut self, current_state: &Vec<f64>) -> usize {
        let mut rng = rand::thread_rng();
        let action: usize;
        let does_explore: f64 = rng.gen_range(f64::MIN, 1.0);
        let max_q_value = self.main_network.generate_q_value(current_state).1; 
        let all_q_values = self.main_network.generate_q_value(current_state).0;
        self.main_network.iterations_passed += 1;
        let epsilon: f64;
        if max_q_value != 0.0 {
            epsilon = 1.0 - (max_q_value)
        } else {
            epsilon = 0.0
        }

        if does_explore < epsilon && self.unexplored_actions.len() != 0{
            action = rng.gen_range(0usize, all_q_values.len() - 1);
            self.unexplored_actions.drain(action..action);
        } else {
            action = all_q_values.iter().position(|&r| r == max_q_value).expect("Invalid max_q_value") as usize
        }
        return action
    }
}