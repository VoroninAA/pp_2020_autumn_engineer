// Copyright 2020 Voronin Aleksey
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include "../../../modules/task_2/voronin_a_vertical_gaussian_method/vertical_gaussian_method.h"

std::vector<double> getRandomMatrixLinear(const int matrixSize) {
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  std::vector<double> linearMatrix(matrixSize * (matrixSize + 1));
  for (int i = 0; i < matrixSize * (matrixSize + 1); i++) {
    linearMatrix[i] = (gen() % 5) + 1;
  }
  return linearMatrix;
}

std::vector<double> sequentialGaussianMethod(std::vector<double> initialMatrix, int equationAmount) {
  if (initialMatrix.size() != (unsigned int)((equationAmount + 1) * equationAmount) || equationAmount <= 0) {
      std::vector<double> empty(0);
      return empty;
  }

  int i, j, k;
  double tmp;
  std::vector<double> results = std::vector<double>(equationAmount);

  for (i = 0; i < equationAmount; i++) {
    tmp = initialMatrix[i * (equationAmount + 1) + i];
    for (j = equationAmount; j >= i; j--)
      initialMatrix[i * (equationAmount + 1) + j] /= tmp;
    for (j = i + 1; j < equationAmount; j++) {
      tmp = initialMatrix[j * (equationAmount + 1) + i];
      for (k = equationAmount; k >= i; k--)
        initialMatrix[j * (equationAmount + 1) + k] -= tmp * initialMatrix[i * (equationAmount + 1) + k];
    }
  }

  results[equationAmount - 1] = initialMatrix[(equationAmount - 1) * (equationAmount + 1) + equationAmount];
  for (i = equationAmount - 2; i >= 0; i--) {
    results[i] = initialMatrix[i * (equationAmount + 1) + equationAmount];
    for (j = i + 1; j < equationAmount; j++)
      results[i] -= initialMatrix[i * (equationAmount + 1) + j] * results[j];
  }

  return results;
}

std::vector<double> parallelGaussianMethod(std::vector<double> initialMatrix, int equationAmount) {
  if (initialMatrix.size() != (unsigned int)((equationAmount + 1) * equationAmount) || equationAmount <= 0) {
      std::vector<double> empty(0);
      return empty;
  }
  std::vector<double> results = std::vector<double>(equationAmount);

  int i, j, k;
  std::vector<int> map(equationAmount);
  double tmp[equationAmount], sum = 0.0;
  int rank, nprocs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  for (i = 0; i < equationAmount; i++)
      map[i] = i % nprocs;

  for (k = 0; k < equationAmount; k++) {
    MPI_Bcast(&initialMatrix[k * (equationAmount + 1) + k], equationAmount - k + 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    for (i = k + 1; i < equationAmount; i++) {
      if (map[i] == rank) {
        tmp[i] = initialMatrix[i * (equationAmount + 1) + k] / initialMatrix[k * (equationAmount + 1) + k];
        for (j = 0; j < equationAmount; j++) {
          initialMatrix[i * (equationAmount + 1) + j] =
          initialMatrix[i * (equationAmount + 1) + j] - (tmp[i] * initialMatrix[k * (equationAmount + 1) + j]);
        }
        initialMatrix[i * (equationAmount + 1) + equationAmount] =
        initialMatrix[i * (equationAmount + 1) + equationAmount] -
        (tmp[i] * initialMatrix[k * (equationAmount + 1) + equationAmount]);
      }
    }
  }

  if (rank == 0) {
    results[equationAmount - 1] =
    initialMatrix[(equationAmount - 1) * (equationAmount + 1) + equationAmount]
    / initialMatrix[(equationAmount - 1) * (equationAmount + 1) + equationAmount - 1];
    for (i = equationAmount - 2; i >= 0; i--) {
      sum = 0;

      for (j = i + 1; j < equationAmount; j++) {
        sum = sum + initialMatrix[i * (equationAmount + 1) + j] * results[j];
      }
      results[i] =
      (initialMatrix[i * (equationAmount + 1) + equationAmount] - sum) / initialMatrix[i * (equationAmount + 1) + i];
    }
  }

  return results;
}
