#include "mpi_advance.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int size = 4096;
    double clusterStartTime = MPI_Wtime();
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* times = networkDiscovery(rank, num_procs, 1000, size);
    double* adjacencyMatrix = (double*) malloc(num_procs * num_procs * sizeof(double));
    MPI_Allgather(times, num_procs, MPI_DOUBLE, adjacencyMatrix, num_procs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    free(times);

    if (rank == 0)
    {
        printf("Adjacency Matrix:\n");
        for (int i = 0; i < num_procs; i++)
        {
            for (int j = 0; j < num_procs; j++)
            {
                printf(".%10f\t", adjacencyMatrix[i * num_procs + j]);
            }
            
            printf("\n");
        }
    }

    int* clusterSizes;
    int* centers;
    int numClusters;
    int** clusters = fullCommunicatorCluster(adjacencyMatrix, rank, num_procs, &clusterSizes, &centers, &numClusters);
    double clusteringTime = MPI_Wtime() - clusterStartTime;
    double finalClusteringTime;
    MPI_Allreduce(&clusteringTime, &finalClusteringTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Clustering took %.10f, message size: %d\n", finalClusteringTime, size);
    }

    MPI_Group worldGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

    MPI_Group localGroup;
    MPI_Group_incl(worldGroup, clusterSizes[0], clusters[0], &localGroup);

    MPI_Comm localComm;
    MPI_Comm_create(MPI_COMM_WORLD, localGroup, &localComm);

    int localRank;
    MPI_Comm_rank(localComm, &localRank);

    printf("Rank: %d, Local rank: %d\n", rank, localRank);
    char* sendbuf = (char*) malloc(size * sizeof(char));
    char* recvbuf = (char*) malloc(size * sizeof(char));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(sendbuf, recvbuf, size, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < 1000; i++)
    {
        MPI_Allreduce(sendbuf, recvbuf, size, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);
    }
    double tfinal = (MPI_Wtime() - t0) / 1000;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("PMPI: %e\n", t0);
    }

    MPI_Comm leaderComm;
    MPI_Comm_split(MPI_COMM_WORLD, localRank, rank, &leaderComm);

    char* local_recv_buf;
    if (localRank == 0)
    {
        local_recv_buf = (char*) malloc(size * sizeof(char));
    }
    else
    {
        local_recv_buf = (char*) malloc(sizeof(char));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(sendbuf, local_recv_buf, size, MPI_CHAR, MPI_MAX, 0, localComm);
    
    if (rank == 0)
    {
        MPI_Allreduce(local_recv_buf, recvbuf, size, MPI_CHAR, MPI_MAX, leaderComm);
    }

    MPI_Bcast(recvbuf, size, MPI_CHAR, 0, localComm);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < 1000; i++)
    {
        MPI_Reduce(sendbuf, local_recv_buf, size, MPI_CHAR, MPI_MAX, 0, localComm);
        
        if (rank == 0)
        {
            MPI_Allreduce(local_recv_buf, recvbuf, size, MPI_CHAR, MPI_MAX, leaderComm);
        }

        MPI_Bcast(recvbuf, size, MPI_CHAR, 0, localComm);
    }
    tfinal = (MPI_Wtime() - t0) / 1000;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("Multileader (%d leaders): %e\n", (num_procs / clusterSizes[0]), t0);
    }

    free(clusterSizes);
    free(centers);
    for (int i = 0; i < numClusters; i++)
    {
        free(clusters[i]);
    }

    free(clusters);

    free(sendbuf);
    free(recvbuf);
    free(local_recv_buf);

    MPI_Finalize();
    return 0;
}