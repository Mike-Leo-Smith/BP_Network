//
// Created by Mike Smith on 2017/3/30.
//

#include <cmath>
#include "sigmoid.h"

namespace bp
{
	double Sigmoid::operator()(double x) const
	{
		return (1.0 / (1.0 + exp(-x)));
	}
	
	double Sigmoid::derivative(double x) const
	{
		double s = (*this)(x);
		return s * (1 - s);
	}
	
	const Eigen::VectorXd Sigmoid::operator()(const Eigen::VectorXd &v) const
	{
		Eigen::VectorXd result(v);
		
		for (size_t i = 0; i < v.size(); i++)
		{
			result[i] = (*this)(result[i]);
		}
		return result;
	}
	
	const Eigen::VectorXd Sigmoid::derivative(const Eigen::VectorXd &v) const
	{
		Eigen::VectorXd result(v);
		
		for (size_t i = 0; i < v.size(); i++)
		{
			result[i] = derivative(result[i]);
		}
		return result;
	}
}