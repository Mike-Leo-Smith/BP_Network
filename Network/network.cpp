//
// Created by Mike Smith on 2017/3/31.
//

#include "network.h"

namespace bp
{
	Network::Network(const std::vector<size_t> &sizes_of_layers)
	{
		for (size_t i = 1; i < sizes_of_layers.size(); i++)
		{
			_layers.push_back(new Layer<Sigmoid>(sizes_of_layers[i - 1], sizes_of_layers[i]));
		}
	}
	
	double Network::predict(const Eigen::VectorXd &input_vector) const
	{
		Eigen::VectorXd result(input_vector);
		
		for (const BaseLayer *layer : _layers)
		{
			result = layer->predict(result);
		}
		
		int result_label = -1;
		double result_label_confidence = -1.0;
		
		for (int i = 0; i < result.size(); i++)
		{
			if (result[i] > result_label_confidence)
			{
				result_label_confidence = result[i];
				result_label = i;
			}
		}
		
		return result_label;
	}
	
	void Network::train(const std::vector<Data> &training_data, const std::vector<Data> test_data, size_t epochs, size_t mini_batch_size, double learning_rate)
	{
		std::vector<Data> data = training_data;
		
		for (size_t i = 0; i < epochs; i++)
		{
			std::random_shuffle(data.begin(), data.end());
			
			for (size_t j = 0; j < data.size(); j += mini_batch_size)
			{
				for (BaseLayer *layer : _layers)
				{
					layer->clear_accumulated_nabla();
				}
				
				// Train a mini batch.
				for (size_t k = 0; k < mini_batch_size; k++)
				{
					Eigen::VectorXd result(data[j + k].data());
					
					for (BaseLayer *layer : _layers)
					{
						layer->feedforward(result);
						result = layer->activation();
					}
					
					Eigen::VectorXd expected(10);
					expected.setZero();
					expected[data[j + k].label()] = 1;
					_layers.back()->backpropagate(expected);
					for (long l = _layers.size() - 2; l >= 0; l--)
					{
						_layers[l]->backpropagate(*_layers[l + 1]);
					}
				}
				
				for (BaseLayer *layer : _layers)
				{
					layer->gradient_descent(mini_batch_size, learning_rate / sqrt(0.1 * i + 1));
				}
			}
			
			int correct_count = 0;
			
			for (const Data &item : test_data)
			{
				if (predict(item.data()) == item.label())
				{
					correct_count++;
				}
			}
			
			double correctness = 1.0 * correct_count / test_data.size() * 100.0;
			std::cout << "Epoch #" << i << " " << "\t\tCorrectness: " << correctness << "%\t\tLearning rate: " << learning_rate / sqrt(0.1 * i + 1) <<std::endl;
		}
	}
}