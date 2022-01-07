use serde::Serialize;
use rayon;

#[derive(Clone, Serialize)]
pub struct ConvLayer {
    convolution_layer: Vec<Vec<f64>>,
    bias: f64,
    trainable: bool
}

impl ConvLayer {
    pub fn initialize(feature_detectors: Vec<Vec<f64>>, trainable: bool) -> Self {
        Self {
            convolution_layer: feature_detectors,
            bias: 0f64,
            trainable: trainable
        }
    }

    pub fn convolve_image(&self, input_image: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut pos_y_filter_layer_end = self.convolution_layer.len();
        let mut pos_y_filter_layer_begin = 0;
        let mut feature_map: Vec<Vec<f64>> = Vec::new();
        let num_threads = input_image[0].len();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads.clone()).build().unwrap();
        while pos_y_filter_layer_end <= num_threads {
            pool.install(|| self.calculate_sum(pos_y_filter_layer_begin, pos_y_filter_layer_end, input_image.clone(), &mut feature_map));
            pos_y_filter_layer_begin += 1;
            pos_y_filter_layer_end += 1;
        }
        feature_map
    }

    pub fn adjust(&mut self, actual_q_value: f64, predicted_value: f64, reward: f64) {
        let delta = (actual_q_value - predicted_value).powf(2.0) * reward;
        for i in 0..self.convolution_layer.len() {
            for j in 0..self.convolution_layer[i].len() {
                self.convolution_layer[i][j] += delta
            }
        }
    }

    fn calculate_sum(&self, pos_y_start: usize, pos_y_end: usize, input_image: Vec<Vec<f64>>, feature_map: &mut Vec<Vec<f64>>) {
        let mut layer: Vec<f64> = Vec::new();
        let mut pos_x_start = 0;
        let mut pos_x_end = self.convolution_layer[0].len();
        while pos_x_end <= input_image[0].len() {
            let mut weighted_sum: f64 = 0f64;
            for i in pos_x_start..pos_x_end {
                for j in pos_y_start..pos_y_end {
                    weighted_sum += self.convolution_layer[j-pos_x_start][i - pos_y_start] * input_image[i][j]
                }
            }
            weighted_sum += self.bias;
            if weighted_sum < 0.0 {
                weighted_sum = 0.0;
            }
            layer.push(weighted_sum);
            pos_x_start += 1;
            pos_x_end += 1;
        }

        feature_map.push(layer);
    }

    fn average_pool(&self, input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut pos_y_max = 1;
        let mut pos_y_min = 0;
        let mut pos_x_max = 1;
        let mut pos_x_min = 0;
        let mut pooled: Vec<Vec<f64>> = Vec::new();
        while pos_y_max < input.len() {
            let mut row: Vec<f64> = Vec::new();
            while pos_x_max < input[0].len(){
                row.push((input[pos_y_min][pos_x_min] + input[pos_y_min][pos_x_max] + input[pos_y_max][pos_x_min] + input[pos_y_max][pos_x_max])/4f64);
                pos_x_max += 2;
                pos_x_min += 2;
            }
            pooled.push(row);
            pos_y_max += 2;
            pos_y_min += 2;
        }
        pooled
    }
}