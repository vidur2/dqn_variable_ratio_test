use crate::dqn::Network;
use rand::Rng;

pub struct Agent {
    target_network: Network,
    main_network: Network,
    replay_buffer: Vec<BufferItem>,
    unexplored_actions: Vec<u64>
}

struct BufferItem {
    current_state: Vec<f64>,
    next_state: Vec<f64>,
    action: usize,
    reward: f64
}

impl Agent {
    pub fn initialize_agent(structure: Vec<u64>, amount_of_states: u64) -> Self {
        let main_network = Network::generate_network(structure.clone(), amount_of_states);
        let target_network = Network::generate_network(structure.clone(), amount_of_states);
        let amount_of_actions = structure[structure.len() - 1];
        let mut unexplored_actions: Vec<u64> = Vec::new();
        for i in 1..amount_of_actions + 1 {
            unexplored_actions.push(i);
        }
        Self {
            target_network: target_network,
            main_network: main_network,
            replay_buffer: Vec::new(),
            unexplored_actions: unexplored_actions
        }
    }

    pub fn act(&mut self, current_state: Vec<f64>, next_state: Vec<f64>, reward: f64){
        let action = self.exploit_explore(&current_state);
        let item = BufferItem {
            current_state: current_state,
            next_state: next_state, 
            action: action,
            reward: reward
        };
        self.replay_buffer.push(item);
    }

    fn exploit_explore(&mut self, current_state: &Vec<f64>) -> usize{
        let mut rng = rand::thread_rng();
        let action: usize;
        let does_explore: f64 = rng.gen_range(f64::MIN..1.0);
        let max_q_value = self.main_network.generate_q_value(current_state).1; 
        let all_q_values = self.main_network.generate_q_value(current_state).0;
        self.main_network.iterations_passed += 1;
        let epsilon: f64;
        if max_q_value != 0.0 {
            let sum: f64 = all_q_values.iter().sum();
            epsilon = 1.0 - (max_q_value/sum)
        } else {
            epsilon = 0.0
        }

        if does_explore < epsilon {
            action = rng.gen_range(0usize..all_q_values.len() - 1);
        } else {
            action = all_q_values.iter().position(|&r| r == max_q_value).expect("Invalid max_q_value") as usize
        }
        return action
    }
}