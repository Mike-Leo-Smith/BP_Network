#include <iostream>
#include <eigen3/Eigen/Core>
#include "Network/network.h"

int main()
{
	Eigen::VectorXd input(3);
	Eigen::MatrixXd mat(2, 3);
	bp::Network network;
	
	input << 1.0, 2.0, 3.0;
	
	std::cout << input * input.transpose() << std::endl;
	std::cout << network.predict(input) << std::endl;
	
	network.train();
	
	return 0;
}