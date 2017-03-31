//
// Created by Mike Smith on 2017/3/30.
//

#ifndef BP_NETWORK_LAYER_H
#define BP_NETWORK_LAYER_H

#include <eigen3/Eigen/Core>
#include "functor.h"

namespace bp
{
	class BaseLayer
	{
	private:
		Eigen::MatrixXd _weight_matrix;
		Eigen::VectorXd _bias_vector;
		Eigen::VectorXd _error_vector;
		Eigen::VectorXd _input_vector;
		Eigen::VectorXd _weighted_input_vector;
		Eigen::VectorXd _activation;
		Eigen::MatrixXd _accumulated_nabla_weights;
		Eigen::VectorXd _accumulated_nabla_bias;
		virtual const Functor &activation_function(void) const = 0;
		
	public:
		BaseLayer(size_t input_dim, size_t output_dim);
		
		virtual const Eigen::VectorXd &error_vector(void) const { return _error_vector; }
		virtual const Eigen::VectorXd &weighted_input_vector(void) const { return _weighted_input_vector; }
		virtual const Eigen::VectorXd &activation(void) const { return _activation; }
		
		virtual void feedforward(const Eigen::VectorXd &input_vector);
		virtual void backpropagate(const BaseLayer &next_layer);
		virtual void backpropagate(const Eigen::VectorXd &expected);
		virtual void gradient_descent(size_t mini_batch_size, double learning_rate);
		virtual void clear_accumulated_nabla(void);
		
		virtual const Eigen::VectorXd predict(const Eigen::VectorXd &input_vector) const;
	};
	
	template<typename FUNCTOR> class Layer : public BaseLayer
	{
	private:
		Functor *_activation_function;
		virtual const Functor &activation_function(void) const override { return *_activation_function; }
	
	public:
		Layer(size_t input_dim, size_t output_dim) : _activation_function(new FUNCTOR), BaseLayer(input_dim, output_dim) {}
		~Layer(void) { delete _activation_function; }
	};
}

#endif  // BP_NETWORK_LAYER_H
