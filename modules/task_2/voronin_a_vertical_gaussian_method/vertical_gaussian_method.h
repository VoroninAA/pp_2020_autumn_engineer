// Copyright 2020 Voronin Aleksey
#ifndef MODULES_TASK_2_VORONIN_A_VERTICAL_GAUSSIAN_METHOD_VERTICAL_GAUSSIAN_METHOD_H_
#define MODULES_TASK_2_VORONIN_A_VERTICAL_GAUSSIAN_METHOD_VERTICAL_GAUSSIAN_METHOD_H_

#include <vector>
std::vector<double> getRandomMatrixLinear(const int matrixSize);
std::vector<double> sequentialGaussianMethod(std::vector<double> initialMatrix, int equationAmount);
// std::vector<double> parallelGaussianMethod(std::vector<double> initialMatrix, int equationAmount);
std::vector <double> parallelGaussianMethod(const std::vector <double> &a, size_t rows, size_t cols);
#endif  // MODULES_TASK_2_VORONIN_A_VERTICAL_GAUSSIAN_METHOD_VERTICAL_GAUSSIAN_METHOD_H_
