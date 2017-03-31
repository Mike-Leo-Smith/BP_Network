//
// Created by Mike Smith on 2017/3/31.
//

#include "functor.h"

namespace bp
{
	const Eigen::VectorXd Functor::derivative(const Eigen::VectorXd &v) const
	{
		Eigen::VectorXd result(v);
		
		for (size_t i = 0; i < v.size(); i++)
		{
			result[i] = derivative(result[i]);
		}
		return result;
	}
	
	const Eigen::VectorXd Functor::operator()(const Eigen::VectorXd &v) const
	{
		Eigen::VectorXd result(v);
		
		for (size_t i = 0; i < v.size(); i++)
		{
			result[i] = (*this)(result[i]);
		}
		return result;
	}
}