//
// Created by Mike Smith on 2017/3/31.
//

#include "mnist_loader.h"
#include "data.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, vector<int, allocator<int>> &labels)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back(label + 0);
		}
		
	}
}

void read_Mnist_Images(string filename, vector<vector<double>>&images)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
		
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;
		
		for (int i = 0; i < number_of_images; i++)
		{
			vector<double> tp;
			
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					
					file.read((char*)&image, sizeof(image));
					tp.push_back((double)image / 255.0);
				}
			}
			images.push_back(tp);
		}
	}
}

void load_mnist(std::vector<bp::Data> &training_data, std::vector<bp::Data> &test_data)
{
	std::vector<std::vector<double>> training_images;
	std::vector<int> training_labels;
	std::vector<std::vector<double>> test_images;
	std::vector<int> test_labels;
	
	read_Mnist_Images("MNIST_DATA/train-images.idx3-ubyte", training_images);
	read_Mnist_Label("MNIST_DATA/train-labels.idx1-ubyte", training_labels);
	
	for (int i = 0; i < training_images.size(); i++)
	{
		Eigen::VectorXd image(training_images[i].size());
		
		for (int j = 0; j < training_images[i].size(); j++)
		{
			image[j] = training_images[i][j];
		}
		training_data.push_back(bp::Data(image, training_labels[i]));
	}
	
	read_Mnist_Images("MNIST_DATA/t10k-images.idx3-ubyte", test_images);
	read_Mnist_Label("MNIST_DATA/t10k-labels.idx1-ubyte", test_labels);
	
	for (int i = 0; i < test_images.size(); i++)
	{
		Eigen::VectorXd image(test_images[i].size());
		
		for (int j = 0; j < test_images[i].size(); j++)
		{
			image[j] = test_images[i][j];
		}
		test_data.push_back(bp::Data(image, test_labels[i]));
	}
}