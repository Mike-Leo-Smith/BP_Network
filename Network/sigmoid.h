//
// Created by Mike Smith on 2017/3/30.
//

#ifndef BP_NETWORK_SIGMOID_H
#define BP_NETWORK_SIGMOID_H

#include "functor.h"

namespace bp
{
	class Sigmoid : public Functor
	{
	public:
		virtual double operator()(double x) const override;
		virtual double derivative(double x) const override;
		virtual const Eigen::VectorXd operator()(const Eigen::VectorXd &v) const override;
		virtual const Eigen::VectorXd derivative(const Eigen::VectorXd &v) const override;
	};
}

#endif  // BP_NETWORK_SIGMOID_H
