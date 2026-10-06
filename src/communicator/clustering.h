#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "mpix_comm.h"

double* networkDiscovery(int rnak, int num_procs, int num_iterations);

int* mahalanobisCluster(double* adjacencyMatrix, int rank, int num_procs, int* clusterSize, int* center);

#endif // CLUSTERING_H