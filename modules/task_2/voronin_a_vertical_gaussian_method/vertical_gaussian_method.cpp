// Copyright 2020 Voronin Aleksey
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include "../../../modules/task_2/voronin_a_vertical_gaussian_method/vertical_gaussian_method.h"

std::vector<double> transMatrix(std::vector<double> matrix, int equationAmount)
{
  double t;
  for (int i = 0; i < equationAmount; i++)
  {
    for (int j = i; j < equationAmount; j++)
    {
      t = matrix[i * (equationAmount + 1) + j];
      matrix[i * (equationAmount + 1) + j] = matrix[j * (equationAmount + 1) + i];
      matrix[j * (equationAmount + 1) + i] = t;
    }
  }
  return matrix;
}

std::vector<double> getRandomMatrixLinear(const int matrixSize)
{
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  std::vector<double> linearMatrix(matrixSize * (matrixSize + 1));
  for (int i = 0; i < matrixSize * (matrixSize + 1); i++)
  {
    linearMatrix[i] = (gen() % 5) + 1;
  }
  return linearMatrix;
}

std::vector<double> sequentialGaussianMethod(std::vector<double> initialMatrix, int equationAmount)
{
  if (initialMatrix.size() != (unsigned int)((equationAmount + 1) * equationAmount) || equationAmount <= 0)
  {
    std::vector<double> empty(0);
    return empty;
  }

  int i, j, k;
  double tmp;
  std::vector<double> results = std::vector<double>(equationAmount);

  for (i = 0; i < equationAmount; i++)
  {
    tmp = initialMatrix[i * (equationAmount + 1) + i];
    for (j = equationAmount; j >= i; j--)
      initialMatrix[i * (equationAmount + 1) + j] /= tmp;
    for (j = i + 1; j < equationAmount; j++)
    {
      tmp = initialMatrix[j * (equationAmount + 1) + i];
      for (k = equationAmount; k >= i; k--)
        initialMatrix[j * (equationAmount + 1) + k] -= tmp * initialMatrix[i * (equationAmount + 1) + k];
    }
  }

  results[equationAmount - 1] = initialMatrix[(equationAmount - 1) * (equationAmount + 1) + equationAmount];
  for (i = equationAmount - 2; i >= 0; i--)
  {
    results[i] = initialMatrix[i * (equationAmount + 1) + equationAmount];
    for (j = i + 1; j < equationAmount; j++)
      results[i] -= initialMatrix[i * (equationAmount + 1) + j] * results[j];
  }

  return results;
}

// std::vector<double> parallelGaussianMethod(std::vector<double> initialMatrix, int equationAmount) {
//   if (initialMatrix.size() != (unsigned int)((equationAmount + 1) * equationAmount) || equationAmount <= 0) {
//       std::vector<double> empty(0);
//       return empty;
//   }
//   std::vector<double> results = std::vector<double>(equationAmount);

//   int i, j, k;
//   std::vector<int> map(equationAmount);
//   std::vector<double> tmp(equationAmount);
//   double sum = 0.0;
//   int rank, nprocs;

//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

//   for (i = 0; i < equationAmount; i++)
//       map[i] = i % nprocs;

//   for (k = 0; k < equationAmount; k++) {
//     MPI_Bcast(&initialMatrix[k * (equationAmount + 1) + k],
// equationAmount - k + 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
//     for (i = k + 1; i < equationAmount; i++) {
//       if (map[i] == rank) {
//         tmp[i] = initialMatrix[i * (equationAmount + 1) + k] / initialMatrix[k * (equationAmount + 1) + k];
//         for (j = 0; j < equationAmount; j++) {
//           initialMatrix[i * (equationAmount + 1) + j] =
//           initialMatrix[i * (equationAmount + 1) + j] - (tmp[i] * initialMatrix[k * (equationAmount + 1) + j]);
//         }
//         initialMatrix[i * (equationAmount + 1) + equationAmount] =
//         initialMatrix[i * (equationAmount + 1) + equationAmount] -
//         (tmp[i] * initialMatrix[k * (equationAmount + 1) + equationAmount]);
//       }
//     }
//   }

//   if (rank == 0) {
//     results[equationAmount - 1] =
//     initialMatrix[(equationAmount - 1) * (equationAmount + 1) + equationAmount]
//     / initialMatrix[(equationAmount - 1) * (equationAmount + 1) + equationAmount - 1];
//     for (i = equationAmount - 2; i >= 0; i--) {
//       sum = 0;

//       for (j = i + 1; j < equationAmount; j++) {
//         sum = sum + initialMatrix[i * (equationAmount + 1) + j] * results[j];
//       }
//       results[i] =
//       (initialMatrix[i * (equationAmount + 1) + equationAmount] - sum) / initialMatrix[i * (equationAmount + 1) + i];
//     }
//   }

//   return results;
// }


std::vector <double> parallelGaussianMethod(const std::vector <double> &a, size_t rows, size_t cols) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int delta = cols / size;
    const int rem = cols % size;

    int code = 0;

    if (rows * cols != a.size()) {
        code = 1;
    }
    MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (rows + 1 != cols) {
        code = 2;
    }
    MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);


    std::vector <double> v((delta + (rank < rem ? 1 : 0)) * rows);

    if (rank == 0) {
        for (int proc = size - 1; proc >= 0; --proc) {
            int index = 0;
            for (size_t j = proc; j < cols; j += size) {
                for (size_t i = 0; i < rows; ++i) {
                    v[index++] = a[i * cols + j];
                }
            }
            if (proc > 0) {
                MPI_Send(v.data(), index, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Status stat;
        MPI_Recv(v.data(), v.size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &stat);
    }

    std::vector <double> pivotCol(rows);
    for (size_t row = 0; row < rows; ++row) {
        if (static_cast<int>(row) % size == rank) {
            int index = 0;
            for (size_t i = rows * (row / size); i < rows * (row / size + 1); ++i) {
                pivotCol[index++] = v[i];
            }
            // assert(index == rows);
        }
        MPI_Bcast(pivotCol.data(), rows, MPI_DOUBLE, row % size, MPI_COMM_WORLD);
        double pivotRow = pivotCol[row];
        for (int j = row / size; j < (delta + (rank < rem ? 1 : 0)); ++j) {
            double pivotC = v[j * rows + row];
            for (size_t k = 0; k < rows; ++k) {
                if (k == row) {
                    v[j * rows + k] /= pivotRow;
                } else {
                    v[j * rows + k] -= pivotC * pivotCol[k] / pivotRow;
                }
            }
        }
    }

    if ((cols - 1) % size == (size_t)rank) {
        MPI_Request rq;
        MPI_Isend(v.data() + ((cols - 1) / size) * rows, rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &rq);
    }
    if (rank == 0) {
        v.resize(rows);
        MPI_Status stat;
        MPI_Recv(v.data(), rows, MPI_DOUBLE, (cols - 1) % size, 2, MPI_COMM_WORLD, &stat);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return v;
}
