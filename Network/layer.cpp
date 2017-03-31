//
// Created by Mike Smith on 2017/3/30.
//

#include <iostream>
#include <random>
#include "layer.h"

namespace bp
{
	BaseLayer::BaseLayer(size_t input_dim, size_t output_dim)
	{
		_weight_matrix.setRandom(output_dim, input_dim);
		_bias_vector.setRandom(output_dim);
		_error_vector.setZero(output_dim);
		_input_vector.setZero(input_dim);
		_weighted_input_vector.setZero(output_dim);
		_activation.setZero(output_dim);
		_accumulated_nabla_weights.setZero(output_dim, input_dim);
		_accumulated_nabla_bias.setZero(output_dim);
	}
	
	void BaseLayer::feedforward(const Eigen::VectorXd &input_vector)
	{
		_input_vector = input_vector;
		_weighted_input_vector = _weight_matrix * _input_vector + _bias_vector;
		_activation = activation_function()(_weighted_input_vector);
	}
	
	void BaseLayer::backpropagate(const BaseLayer &next_layer)
	{
		_error_vector = (next_layer._weight_matrix.transpose() * next_layer._error_vector).cwiseProduct(activation_function().derivative(_weighted_input_vector));
		_accumulated_nabla_weights += (_error_vector * _input_vector.transpose());
		_accumulated_nabla_bias += _error_vector;
	}
	
	void BaseLayer::gradient_descent(size_t mini_batch_size, double learning_rate)
	{
		_weight_matrix -= learning_rate / mini_batch_size * _accumulated_nabla_weights;
		_bias_vector -= learning_rate / mini_batch_size * _accumulated_nabla_bias;
	}
	
	void BaseLayer::clear_accumulated_nabla(void)
	{
		_accumulated_nabla_bias.setZero();
		_accumulated_nabla_weights.setZero();
	}
	
	void BaseLayer::backpropagate(const Eigen::VectorXd &expected)
	{
		_error_vector = (_activation - expected).cwiseProduct(activation_function().derivative(_weighted_input_vector));
		_accumulated_nabla_weights += (_error_vector * _input_vector.transpose());
		_accumulated_nabla_bias += _error_vector;
	}
	
	const Eigen::VectorXd BaseLayer::predict(const Eigen::VectorXd &input_vector) const
	{
		return activation_function()(_weight_matrix * input_vector + _bias_vector);
	}
}