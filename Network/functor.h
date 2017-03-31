//
// Created by Mike Smith on 2017/3/30.
//

#ifndef BP_NETWORK_FUNCTOR_H
#define BP_NETWORK_FUNCTOR_H

#include <eigen3/Eigen/Core>

namespace bp
{
	class Functor
	{
	public:
		virtual double operator()(double x) const = 0;
		virtual double derivative(double x) const = 0;
		virtual const Eigen::VectorXd operator()(const Eigen::VectorXd &v) const;
		virtual const Eigen::VectorXd derivative(const Eigen::VectorXd &v) const;
	};
}

#endif  // BP_NETWORK_FUNCTOR_H
