//
// Created by Mike Smith on 2017/3/31.
//

#ifndef BP_NETWORK_NETWORK_H
#define BP_NETWORK_NETWORK_H

#include <vector>
#include <iostream>
#include "layer.h"
#include "sigmoid.h"
#include "data.h"

namespace bp
{
	class Network
	{
	private:
		std::vector<BaseLayer *> _layers;
		
	public:
		Network(const std::vector<size_t> &sizes_of_layers);
		double predict(const Eigen::VectorXd &input_vector) const;
		void train(const std::vector<Data> &training_data, const std::vector<Data> test_data, size_t epochs, size_t mini_batch_size, double learning_rate);
	};
}

#endif  // BP_NETWORK_NETWORK_H
