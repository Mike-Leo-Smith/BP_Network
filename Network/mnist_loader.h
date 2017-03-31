//
// Created by Mike Smith on 2017/3/31.
//

#ifndef BP_NETWORK_MNIST_LOADER_H
#define BP_NETWORK_MNIST_LOADER_H

#include <string>
#include <vector>
#include "data.h"

void read_Mnist_Label(std::string filename, std::vector<double> &labels);
void read_Mnist_Images(std::string filename, std::vector <std::vector<double>> &images);
void load_mnist(std::vector<bp::Data> &training_data, std::vector<bp::Data> &test_data);

#endif  // BP_NETWORK_MNIST_LOADER_H
