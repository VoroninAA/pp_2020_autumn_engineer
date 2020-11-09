// Copyright 2020 Voronin Aleksey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./vertical_gaussian_method.h"

TEST(Parallel_Operations_MPI, can_get_random_linear_matrix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        const int unknownNumber = 5;
        ASSERT_NO_THROW(getRandomMatrixLinear(unknownNumber));
    }
}

TEST(Parallel_Operations_MPI, can_get_result_with_random_matrix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        const int unknownNumber = 5;
        std::vector<double> sampleMatrix = getRandomMatrixLinear(unknownNumber);
        ASSERT_NO_THROW(sequentialGaussianMethod(sampleMatrix, unknownNumber));
    }
}

TEST(Parallel_Operations_MPI, can_get_result_with_sequential_version_three_unknowns) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        const int unknownNumber = 3;
        std::vector<double> sampleMatrix = {2, 1, 1, 2,
                                            1, -1, 0, -2,
                                            3, -1, 2, 2};
        std::vector<double> result = sequentialGaussianMethod(sampleMatrix, unknownNumber);
        std::vector<double> expectedResult = {-1, 1, 3};
        ASSERT_EQ(result, expectedResult);
    }
}

TEST(Parallel_Operations_MPI, can_get_result_with_sequential_version_two_unknowns) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        const int unknownNumber = 2;
        std::vector<double> sampleMatrix = {1, -1, -5,
                                            2, 1, -7};
        std::vector<double> result = sequentialGaussianMethod(sampleMatrix, unknownNumber);
        std::vector<double> expectedResult = {-4, 1};
        ASSERT_EQ(result, expectedResult);
    }
}

TEST(Parallel_Operations_MPI, cant_get_result_with_wrong_input) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        const int unknownNumber = 4;
        std::vector<double> sampleMatrix = {1, -1, -5,
                                            2, 1, -7};
        ASSERT_ANY_THROW(std::vector<double> result = sequentialGaussianMethod(sampleMatrix, unknownNumber));
    }
}

TEST(Parallel_Operations_MPI, cant_get_result_with_negative_input) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        const int unknownNumber = -1;
        std::vector<double> sampleMatrix = {1, -1, -5,
                                            2, 1, -7};
        ASSERT_ANY_THROW(std::vector<double> result = sequentialGaussianMethod(sampleMatrix, unknownNumber));
    }
}

TEST(Parallel_Operations_MPI, can_get_result_with_parallel_version_two_unknowns) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int unknownNumber = 2;
    std::vector<double> sampleMatrix = {1, -1, -5,
                                        2, 1, -7};
    std::vector<double> result = parallelGaussianMethod(sampleMatrix, unknownNumber);
    if (rank == 0) {
        std::vector<double> expectedResult = {-4, 1};
        ASSERT_EQ(result, expectedResult);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
