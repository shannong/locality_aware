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

double pingpong(char* buffer, int rank, int proc, int tag, int n_iter, int size)
{
    double t0 = MPI_Wtime();
    if (rank < proc)
    {
        for (int i = 0; i < n_iter; i++)
        {
            MPI_Send(buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD);
            MPI_Recv(buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (int i = 0; i < n_iter; i++)
        {
            MPI_Recv(buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD);
        }
    }

    double tfinal = (MPI_Wtime() - t0) / n_iter;

    return tfinal;
}

double* network_discovery(int rank, int num_procs, int num_iterations, int size)
{
    char* buf = (char*) malloc(size * sizeof(char));
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
            double t0 = pingpong(buf, rank, proc, tag, 1, size);
            double time = pingpong(buf, rank, proc, tag, num_iterations, size);
            times[proc] = time;
        }

        tag++;
    }

    free(buf);
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
    double meanDistance = mean(procDistances, targetRank);
    double stdDev = standardDeviation(meanDistance, procDistances, targetRank);
    return (procDistances[targetRank].distance - meanDistance) / stdDev; 
}

int compareProcDistances(const void* a, const void* b)
{
    ProcDistance procDistanceA = *(const ProcDistance*) a;
    ProcDistance procDistanceB = *(const ProcDistance*) b;

    if (procDistanceA.distance < procDistanceB.distance)
    {
        return -1;
    }
    else if (procDistanceA.distance > procDistanceB.distance)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int compareInts(const void* a, const void* b)
{
    int intA = *(const int*) a;
    int intB = *(const int*) b;
    if (intA < intB)
    {
        return -1;
    }
    else if (intA > intB)
    {
        return 1;
    }
    else 
    {
        return 0;
    }
}

void buildClusterForRank(int rank, int num_procs, double* rankTimes, int* clusterSize, int** cluster)
{
    ProcDistance procDistances[num_procs];
    int rowStart = rank * num_procs;
    for (int i = rowStart, j = 0; i < rowStart + num_procs && j < num_procs; i++, j++)
    {
        procDistances[j] = (ProcDistance){j, rankTimes[i]};
    }

    qsort(procDistances, num_procs, sizeof(ProcDistance), compareProcDistances);

    *clusterSize = 2;
    double md;
    *cluster = (int*) malloc(num_procs * sizeof(int));
    (*cluster)[0] = rank;
    (*cluster)[1] = procDistances[1].rank;

    do {
        md = mahalanobisDistance(*clusterSize,  procDistances);
        if (md <= 1.96)
        {
            (*cluster)[*clusterSize] = procDistances[*clusterSize].rank;
            (*clusterSize)++;      
        }
    } while (md <= 1.96);

    for (int i = *clusterSize; i < num_procs; i++)
    {
        (*cluster)[i] = INT_MAX;
    }
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

void mahalanobisCluster(double* adjacencyMatrix, int rank, int num_procs, int* clusterSize, int* center, int** cluster)
{
    buildClusterForRank(rank, num_procs, adjacencyMatrix, clusterSize, cluster);

    qsort(*cluster, *clusterSize, sizeof(int), compareInts);

    bool changed;
    do {
        changed = false;
        int additionalRanks[num_procs - (*clusterSize)];
        int numAdditionalRanks = 0;
        for (int i = 0; i < *clusterSize; i++)
        {
            int targetRank = (*cluster)[i];
            if (targetRank != rank)
            {
                int targetClusterSize;
                int* targetCluster;
                buildClusterForRank(targetRank, num_procs, adjacencyMatrix, &targetClusterSize, &targetCluster);
                for (int j = 0; j < targetClusterSize; j++)
                {
                    int targetClusterRank = targetCluster[j];
                    if (!isRankInCluster(targetClusterRank, *cluster, *clusterSize))
                    {
                        additionalRanks[numAdditionalRanks] = targetClusterRank;
                        numAdditionalRanks++;
                    }
                }

                free(targetCluster);
            }

        }

        if (numAdditionalRanks != 0)
        {
            changed = true;
            for (int i = 0; i < numAdditionalRanks; i++)
            {
                (*cluster)[*clusterSize] = additionalRanks[i];
                clusterSize++;
            }

            qsort(cluster, *clusterSize, sizeof(int), compareInts);
        }
    } while (changed && *clusterSize < num_procs);

    *center = findCenter(*cluster, *clusterSize, adjacencyMatrix, num_procs);

    int* temp = (int*) realloc(*cluster, (*clusterSize) * sizeof(int));
    if (temp != NULL)
    {
        *cluster = temp;
    }
}

void extendCluster(int rank, int numProcs, int startingPoint, ProcDistance* procDistances, int** cluster, int* clusterSize)
{
    *cluster = (int*) malloc(numProcs * sizeof(int));
    for (int i = 0; i < startingPoint; i++)
    {
        (*cluster)[i] = procDistances[i].rank;
    }

    *clusterSize = startingPoint;

    double previousMd = mahalanobisDistance(startingPoint, procDistances);
    double currentMd = previousMd;
    while (startingPoint < numProcs && (currentMd <= previousMd || currentMd <= 1.96))
    {
        (*cluster)[startingPoint] = procDistances[startingPoint].rank;
        startingPoint++;
        (*clusterSize)++;
        previousMd = currentMd;
        currentMd = mahalanobisDistance(startingPoint, procDistances);
    }

    int* temp = (int*) realloc(*cluster, (*clusterSize) * sizeof(int));
    if (temp != NULL)
    {
        *cluster = temp;
    }
}

int** fullCommunicatorCluster(double* adjacencyMatrix, int rank, int num_procs, int** clusterSizes, int** centers, int* numClusters)
{
    ProcDistance* procDistances = (ProcDistance*) malloc(num_procs * sizeof(ProcDistance));
    int rowStart = rank * num_procs;
    for (int i = rowStart, j = 0; i < rowStart + num_procs && j < num_procs; i++, j++)
    {
        procDistances[j] = (ProcDistance){j, adjacencyMatrix[i]};
    }

    qsort(procDistances, num_procs, sizeof(ProcDistance), compareProcDistances);

    int maxNumClusters = (num_procs + 1) / 2;

    int** clusters = (int**) malloc(maxNumClusters * sizeof(int*));
    *clusterSizes = (int*) malloc(maxNumClusters * sizeof(int));

    int startingPoint = 0;
    *numClusters = 1;

    mahalanobisCluster(adjacencyMatrix, rank, num_procs, clusterSizes[0], centers[0], clusters);
    startingPoint = (*clusterSizes)[0];

    while (startingPoint < num_procs)
    {
        extendCluster(rank, num_procs, startingPoint, procDistances, &clusters[*numClusters], &(*clusterSizes)[*numClusters]);
        startingPoint += (*clusterSizes)[*numClusters];
        (*numClusters)++;
    }

    int** tempClusters = (int**) realloc(clusters, (*numClusters) * sizeof(int*));
    if (tempClusters != NULL)
    {
        clusters = tempClusters;
    }

    int* tempSizes = (int*) realloc(*clusterSizes, (*numClusters) * sizeof(int));
    if (tempSizes != NULL)
    {
        *clusterSizes = tempSizes;
    }

    free(procDistances);

    return clusters;
}