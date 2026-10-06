#include "clustering.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>


struct ProcDistance {
    int rank;
    double distance;
};

int mod(int a, int m)
{
    return ((a % m) + m) % m;
}

double pingpong(float* buffer, int rank, int proc, int tag, int n_iter)
{
    double t0 = MPI_Wtime();
    if (rank < proc)
    {
        for (int i = 0; i < n_iter; i++)
        {
            MPI_Send(buffer, 1, MPI_FLOAT, proc, tag, MPI_COMM_WORLD);
            MPI_Recv(buffer, 1, MPI_FLOAT, proc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (int i = 0; i < n_iter; i++)
        {
            MPI_Recv(buffer, 1, MPI_FLOAT, proc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer, 1, MPI_FLOAT, proc, tag, MPI_COMM_WORLD);
        }
    }

    double tfinal = (MPI_Wtime() - t0) / n_iter;
    
    return tfinal;
}

double* network_discovery(int rank, int num_procs, int num_iterations)
{
    float buf;
    int proc, pos;
    int tag = 0;
    
    int n = num_procs;
    double times[num_procs];
    times[rank] = 0.0;

    if (num_procs % 2 == 1)
    {
        n++;
    }

    for (int r = 1; r < n; r++)
    {
        if (rank == 0)
        {
            proc = r;
        } 
        else if (rank == r)
        {
            proc = 0;
        }
        else
        {
            int i = rank;
            pos = 2 * r - i;
            int j = mod(pos, n - 1);
            if (j == 0) 
            {
                j = n - 1;
            }

            proc = j;
        }

        if (proc < num_procs)
        {
            double t0 = pingpong(&buf, rank, proc, tag, 1);
            double time = pingpong(&buf, rank, proc, tag, num_iterations);
            times[proc] = time;
        }

        tag++;
    }

    return times;
}

double mean(const ProcDistance* data, int count)
{
    double sum = 0.0;
    for (int i = 0; i < count; i++)
    {
        sum += data[i].distance;
    }

    return sum / (double) count;
}

double standardDeviation(double mean, const ProcDistance* data, int count)
{
    double sumSquares = 0.0;   
    for (int i = 0; i < count; i++)
    {
        double difference = data[i].distance - mean;
        sumSquares += difference * difference;
    }

    return sqrt(sumSquares / (double) (count - 1));
}

double mahalanobisDistance(int targetRank, const ProcDistance* procDistances)
{
    double meanDistance = mean(procDistances, targetRank - 1);
    double stdDev = standardDeviation(meanDistance, procDistances, targetRank - 1);
    return (procDistances[targetRank - 1].distance - meanDistance) / stdDev; 
}

int compareProcDistances(const void* a, const void* b)
{
    ProcDistance procDistanceA = *(const ProcDistance*) a;
    ProcDistance procDistanceB = *(const ProcDistance*) b;

    return procDistanceA.distance - procDistanceB.distance;
}

int compareInts(const void* a, const void* b)
{
    int intA = *(const int*) a;
    int intB = *(const int*) b;
    return intA - intB;
}

int* buildClusterForRank(int rank, int num_procs, double* rankTimes, int* clusterSize)
{
    ProcDistance procDistances[num_procs];
    int rowStart = rank * num_procs;
    for (int i = rowStart; i < rowStart + num_procs; i++)
    {
        procDistances[i] = {i, rankTimes[i]};
    }

    qsort(procDistances, num_procs, sizeof(ProcDistance), compareProcDistances);

    *clusterSize = 1;
    double md;
    int cluster[num_procs];
    cluster[0] = rank;

    int i = 1;
    do {
        md = mahalanobisDistance(i,  procDistances);
        cluster[*clusterSize] = procDistances[i].rank;
        *clusterSize++;      
        i++;
    } while (md <= 1.96);

    for (int i = *clusterSize; i < num_procs; i++)
    {
        cluster[i] = INT_MAX;
    }

    return cluster;
}

// Assumes cluster is sorted
bool isRankInCluster(int rank, int* cluster, int clusterSize)
{
    int low = 0;
    int high = clusterSize - 1;
    if (rank < cluster[0] || rank > cluster[clusterSize - 1])
    {
        return false;
    }

    while (low <= high)
    {
        int mid = low + (high - low) / 2;
        if (cluster[mid] == rank)
        {
            return true;
        }

        if (cluster[mid] < rank)
        {
            low = mid + 1;
        }
        else 
        {
            high = mid - 1;
        }
    }

    return false;
}

int findCenter(int* cluster, int clusterSize, double* rankTimes, int num_procs)
{
    double minEnergy = DBL_MAX;
    int center = -1;
    for (int i = 0; i < clusterSize; i++)
    {
        double energy = 0.0;
        for (int j = 0; j < clusterSize; j++)
        {
            double distance = rankTimes[i * num_procs + j];
            energy += distance * distance;
        }

        if (energy < minEnergy)
        {
            minEnergy = energy;
            center = i;
        }
    }

    return center;
}

// TODO: figure out a hardcoded number of iterations?
int* mahalanobisCluster(double* adjacencyMatrix, int rank, int num_procs, int* clusterSize, int* center)
{
    int* cluster = buildClusterForRank(rank, num_procs, adjacencyMatrix, clusterSize);

    qsort(cluster, *clusterSize, sizeof(int), compareInts);

    bool changed;
    do {
        changed = false;
        int additionalRanks[num_procs - *clusterSize];
        int numAdditionalRanks = 0;
        for (int i = 0; i < *clusterSize; i++)
        {
            int targetRank = cluster[i];
            int targetClusterSize;
            int* targetCluster = buildClusterForRank(targetRank, num_procs, adjacencyMatrix, &targetClusterSize);

            for (int j = 0; j < targetClusterSize; j++)
            {
                int targetClusterRank = targetCluster[j];
                if (!isRankInCluster(targetClusterRank, cluster, *clusterSize))
                {
                    additionalRanks[numAdditionalRanks] = targetClusterRank;
                    numAdditionalRanks++;
                }
            }
        }

        if (numAdditionalRanks != 0)
        {
            changed = true;
            for (int i = 0; i < numAdditionalRanks; i++)
            {
                cluster[*clusterSize] = additionalRanks[i];
                clusterSize++;
            }

            qsort(cluster, *clusterSize, sizeof(int), compareInts);
        }
    } while (changed && *clusterSize < num_procs);

    *center = findCenter(cluster, *clusterSize, adjacencyMatrix, num_procs);
}