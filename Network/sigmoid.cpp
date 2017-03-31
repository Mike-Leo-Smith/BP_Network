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
}