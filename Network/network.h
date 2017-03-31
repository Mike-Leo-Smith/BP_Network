//
// Created by Mike Smith on 2017/3/31.
//

#ifndef BP_NETWORK_NETWORK_H
#define BP_NETWORK_NETWORK_H

#include <vector>
#include "layer.h"
#include "sigmoid.h"

namespace bp
{
	class Network
	{
	private:
		std::vector<BaseLayer *> _layers;
		
	public:
		Network(void)
		{
			_layers.push_back(new Layer<Sigmoid>(3, 5));
			_layers.push_back(new Layer<Sigmoid>(5, 7));
			_layers.push_back(new Layer<Sigmoid>(7, 10));
		}
		
		const Eigen::VectorXd predict(const Eigen::VectorXd &input_vector) const
		{
			Eigen::VectorXd result(input_vector);
			
			for (const BaseLayer *layer : _layers)
			{
				result = layer->predict(result);
			}
			return result;
		}
		
		void train(void)
		{
			Eigen::VectorXd input(3);
			Eigen::VectorXd expected(10);
			
			input.setRandom();
			expected.setRandom();
			
			for (BaseLayer *layer : _layers)
			{
				layer->feedforward(input);
				input = layer->activation();
			}
			
			_layers[2]->backpropagate(expected);
			_layers[1]->backpropagate(*_layers[2]);
			_layers[0]->backpropagate(*_layers[1]);
		}
	};
}

#endif  // BP_NETWORK_NETWORK_H
