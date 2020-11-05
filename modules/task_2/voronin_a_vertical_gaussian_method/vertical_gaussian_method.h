// Copyright 2020 Voronin Aleksey
#ifndef MODULES_TASK_2_VORONIN_A_VERTICAL_GAUSSIAN_METHOD_H_
#define MODULES_TASK_2_VORONIN_A_VERTICAL_GAUSSIAN_METHOD_H_

#include <vector>

// std::vector<int> getRandomVector(int  size);
// std::vector<int> getSampleMatrix();
std::vector<double> sequentialGaussianMethod(std::vector<double> initialMatrix, int equationAmount);
std::vector<double> parallelGaussianMethod(std::vector<double> initialMatrix, int equationAmount);
// int getParallelOperations(std::vector<int> global_vec, std::vector<int> second_global_vec,
//                           int count_size_vector);
// int getSequentialOperations(std::vector<int> vec, std::vector<int> second_vec);

#endif  // MODULES_TASK_2_VORONIN_A_VERTICAL_GAUSSIAN_METHOD_H_
