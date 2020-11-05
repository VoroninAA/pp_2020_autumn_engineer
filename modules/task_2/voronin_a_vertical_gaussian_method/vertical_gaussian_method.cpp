// Copyright 2020 Voronin Aleksey
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <math.h>
#include "../../../modules/task_2/voronin_a_vertical_gaussian_method/vertical_gaussian_method.h"
using namespace std;

// std::vector<int> getRandomVector(int size) {
//     std::mt19937 gen;
//     gen.seed(static_cast<unsigned int>(time(0)));
//     std::vector<int> vec(size);
//     for (int  i = 0; i < size; i++) { vec[i] = gen() % 10; }
//     return vec;
// }
// std::vector<int> getSampleMatrix(){
//     std::vector<int> vector;
//     return vector;
// }
std::vector<double> sequentialGaussianMethod(std::vector<double> initialMatrix, int equationAmount){
    if(initialMatrix.size()%equationAmount != 0)
        std::__throw_runtime_error("Wrong Input");
    int i, j, k, rowSize=initialMatrix.size()/equationAmount;
    double  tmp;
    std::vector<double> results =std::vector<double>(rowSize-1);

 
 //выводим массив
    cout << "initialMatrix: " << endl;
      for (i=0; i<equationAmount; i++)
       {
          for (j=0; j<rowSize; j++)
            cout << initialMatrix[i*(equationAmount+1)+j] << " ";
        cout << endl;
       }
    cout << endl;
 
//Метод Гаусса
//Прямой ход, приведение к верхнетреугольному виду

 
    for (i=0; i<equationAmount; i++)
     {
       tmp=initialMatrix[i*(equationAmount+1)+i];
         for (j=equationAmount;j>=i;j--)
             initialMatrix[i*(equationAmount+1)+j]/=tmp;
           for (j=i+1;j<equationAmount;j++)
          {
             tmp=initialMatrix[j*(equationAmount+1)+i];
               for (k=equationAmount;k>=i;k--)
             initialMatrix[j*(equationAmount+1)+k]-=tmp*initialMatrix[i*(equationAmount+1)+k];
          }
      }
  /*обратный ход*/
    results[equationAmount-1] = initialMatrix[(equationAmount-1)*(equationAmount+1)+equationAmount];
     for (i=equationAmount-2; i>=0; i--)
       {
           results[i] = initialMatrix[i*(equationAmount+1)+equationAmount];
           for (j=i+1;j<equationAmount;j++) results[i]-=initialMatrix[i*(equationAmount+1)+j]*results[j];
       }
 
return results;
 
}



std::vector<double> parallelGaussianMethod(std::vector<double> initialMatrix, int equationAmount){
    if(initialMatrix.size()%equationAmount != 0)
        std::__throw_runtime_error("Wrong Input");
    int rowSize=initialMatrix.size()/equationAmount;
    std::vector<double> results =std::vector<double>(rowSize-1);
 
  int i,j,k;
    int map[500];
    double A[500][500],b[500],c[500],x[500],sum=0.0;
    double range=1.0;
    int n=3;
    int rank, nprocs;
    clock_t begin1, end1, begin2, end2;
    //MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

    //////////////////////////////////////////////////////////////////////////////////

    if (rank==0)
    {
        for (i=0; i<n; i++)
        {
            for (j=0; j<n; j++)
                A[i][j]=range*(1.0-2.0*(double)rand()/RAND_MAX);
            b[i]=range*(1.0-2.0*(double)rand()/RAND_MAX);
        }
        printf("\n Matrix A (generated randomly):\n");
        for (i=0; i<n; i++)
        {
            for (j=0; j<n; j++)
                printf("%9.6lf ",A[i][j]);
            printf("\n");
        }
        printf("\n Vector b (generated randomly):\n");
        for (i=0; i<n; i++)
            printf("%9.6lf ",b[i]);
        printf("\n\n");
    }

    //////////////////////////////////////////////////////////////////////////////////

    begin1 =clock();

    MPI_Bcast (&A[0][0],500*500,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast (b,n,MPI_DOUBLE,0,MPI_COMM_WORLD);    

    for(i=0; i<n; i++)
    {
        map[i]= i % nprocs;
    } 

    for(k=0;k<n;k++)
    {
        MPI_Bcast (&A[k][k],n-k,MPI_DOUBLE,map[k],MPI_COMM_WORLD);
        MPI_Bcast (&b[k],1,MPI_DOUBLE,map[k],MPI_COMM_WORLD);
        for(i= k+1; i<n; i++) 
        {
            if(map[i] == rank)
            {
                c[i]=A[i][k]/A[k][k];
            }
        }               
        for(i= k+1; i<n; i++) 
        {       
            if(map[i] == rank)
            {
                for(j=0;j<n;j++)
                {
                    A[i][j]=A[i][j]-( c[i]*A[k][j] );
                }
                b[i]=b[i]-( c[i]*b[k] );
            }
        }
    }
    end1 = clock();

    //////////////////////////////////////////////////////////////////////////////////

    begin2 =clock();

    if (rank==0)
    { 
        x[n-1]=b[n-1]/A[n-1][n-1];
        for(i=n-2;i>=0;i--)
        {
            sum=0;

            for(j=i+1;j<n;j++)
            {
                sum=sum+A[i][j]*x[j];
            }
            x[i]=(b[i]-sum)/A[i][i];
        }

        end2 = clock();
    }
    //////////////////////////////////////////////////////////////////////////////////
    if (rank==0)
    { 
        printf("\nThe solution is:");
        for(i=0;i<n;i++)
        {
            printf("\nx%d=%f\t",i,x[i]);

        }

        printf("\n\nLU decomposition time: %f", (double)(end1 - begin1) / CLOCKS_PER_SEC);
        printf("\nBack substitution time: %f\n", (double)(end2 - begin2) / CLOCKS_PER_SEC);
    }
    
    return results;
}

// int getSequentialOperations(std::vector<int> vec, std::vector<int> second_vec) {
//     const int  left = vec.size();
//     int sum = 0;

//         for (int  i = 0; i < left; i++) {
//             sum += vec[i]*second_vec[i];
//         }
//     return sum;
// }

// int getParallelOperations(std::vector<int> global_vec, std::vector<int> second_global_vec,
//                           int count_size_vector) {
//     int size, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     const int delta = count_size_vector / size;

//     if (rank == 0) {
//         int sd = count_size_vector%size;
//         for (int proc = 1; proc < size; proc++) {
//             MPI_Send(&global_vec[0] + sd + proc * delta, delta,
//                         MPI_INT, proc, 0, MPI_COMM_WORLD);
//             MPI_Send(&second_global_vec[0] + sd + proc * delta, delta,
//                         MPI_INT, proc, 1, MPI_COMM_WORLD);
//         }
//     }

//     std::vector<int> local_vec(delta);
//     std::vector<int> second_local_vec(delta);
//     if (rank == 0) {
//         int left = count_size_vector% size;
//         local_vec = std::vector<int>(global_vec.begin(),
//                                      global_vec.begin() + delta+left);
//         second_local_vec = std::vector<int>(second_global_vec.begin(),
//                                      second_global_vec.begin() + delta+left);
//     } else {
//         MPI_Status status;
//         MPI_Recv(&local_vec[0], delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
//         MPI_Recv(&second_local_vec[0], delta, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
//     }

//     int global_sum = 0;
//     int local_sum = getSequentialOperations(local_vec, second_local_vec);
//     MPI_Op op_code;
//      op_code = MPI_SUM;
//     MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, op_code, 0, MPI_COMM_WORLD);
//     return global_sum;
// }
