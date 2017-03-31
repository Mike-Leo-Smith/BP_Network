#include <iostream>
#include <eigen3/Eigen/Core>
#include "Network/network.h"
#include "Network/mnist_loader.h"

int main()
{
	std::vector<bp::Data> training, test;
	std::vector<size_t> sizes;
	
	sizes.push_back(28 * 28);
	sizes.push_back(50);
	sizes.push_back(100);
	sizes.push_back(70);
	sizes.push_back(10);
	
	load_mnist(training, test);
	bp::Network network(sizes);
	network.train(training, test, 300, 20, 1.0);
	
	return 0;
}