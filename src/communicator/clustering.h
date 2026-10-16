#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "mpix_comm.h"

double* networkDiscovery(int rank, int num_procs, int num_iterations);

void mahalanobisCluster(double* adjacencyMatrix, int rank, int num_procs, int* clusterSize, int* center, int** cluster);

int** fullCommunicatorCluster(double* adjacencyMatrix, int rank, int num_procs, int** clusterSizes, int** centers, int* numClusters);

#endif // CLUSTERING_H