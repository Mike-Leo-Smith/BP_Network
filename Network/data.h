//
// Created by Mike Smith on 2017/3/31.
//

#ifndef BP_NETWORK_DATA_H
#define BP_NETWORK_DATA_H

#include <eigen3/Eigen/Core>

namespace bp
{
	class Data
	{
	private:
		Eigen::VectorXd _data;
		int _label;
		
	public:
		Data(const Eigen::VectorXd &data, int label) : _data(data), _label(label) {}
		const Eigen::VectorXd &data(void) const { return _data; }
		int label(void) const { return _label; }
	};
}

#endif  // BP_NETWORK_DATA_H
