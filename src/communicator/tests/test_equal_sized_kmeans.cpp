#include "mpi_advance.h"

#include <cfloat>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <vector>

double* calculateCostMatrix(int numClusters, int* centers, int* procsPerCluster, int numProcs, double* adjacencyMatrix)
{
    double costMatrix[numProcs * numProcs];

    int clusterOffset = 0;
    for (int i = 0; i < numClusters; i++)
    {
        int center = centers[i];
        int clusterSize = procsPerCluster[i];
        for (int j = 0; j < numProcs; j++)
        {
            double distanceToCenter = adjacencyMatrix[j * numProcs + center];
            for (int k = 0; k < clusterSize; k++)
            {
                costMatrix[clusterOffset + k] = distanceToCenter * distanceToCenter;
            }

            clusterOffset += clusterSize;
        }
    }

    return costMatrix;
}

void hungarian(const double* costMatrix, int numProcs, int* colAssignments)
{
    double rowPotentials[numProcs]; // indexed by rows
    double colPotentials[numProcs]; // indexed by cols
    int predecessor[numProcs]; // indexed by cols
    double minSlack[numProcs]; // indexed by cols
    bool visitedRows[numProcs];
    bool visitedCols[numProcs];

    for (int i = 0; i < numProcs; i++)
    {
        rowPotentials[i] = 0.0;
        colAssignments[i] = -1;
        colPotentials[i] = 0.0;
    }

    for (int rootRow = 0; rootRow < numProcs; rootRow++)
    {
        for (int i = 0; i < numProcs; i++)
        {
            predecessor[i] = -1;
            minSlack[i] = DBL_MAX;
            visitedCols[i] = false;
            visitedRows[i] = false;
        }

        int currentRow = rootRow;
        int targetCol = -1;
        while (true)
        {
            visitedRows[currentRow] = true;
            double delta = DBL_MAX;
            int nextCol = -1;

            for (int col = 0; col < numProcs; col++)
            {
                if (!visitedCols[col])
                {
                    double currentSlack = costMatrix[currentRow * numProcs + col] - rowPotentials[currentRow] - colPotentials[col];
                    if (currentSlack < minSlack[col])
                    {
                        minSlack[col] = currentSlack;
                        predecessor[col] = targetCol;
                    }

                    if (minSlack[col] < delta)
                    {
                        delta = minSlack[col];
                        nextCol = col;
                    }
                }
            }

            for (int i = 0; i < numProcs; i++)
            {
                if (visitedRows[i])
                {
                    rowPotentials[i] += delta;
                }

                if (visitedCols[i])
                {
                    colPotentials[i] -= delta;
                }
                else
                {
                    minSlack[i] -= delta;
                }
            }

            targetCol = nextCol;
            visitedCols[targetCol] = true;
            if (colAssignments[targetCol] == -1)
            {
                break;
            }

            currentRow = colAssignments[targetCol];
        }

        while (targetCol != -1)
        {
            int previousCol = predecessor[targetCol];
            int matchedRow = previousCol == -1 ? rootRow : colAssignments[previousCol];
            colAssignments[targetCol] = matchedRow;
            targetCol = previousCol;
        }
    }
}

std::vector<int> initialize(int numNodes, int numSockets, int numProcs)
{
    int numCenters = numNodes * numSockets;
    int clusterSize = numProcs / numCenters;
    std::vector<int> centers;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int>(0, numCenters);
    while (centers.size() < numCenters)
    {
        std::vector<double> distancesSquared;
        for (int i = 0; i < numProcs; i++)
        {
            double minDistance = adjacencyMatrix.at(i * numProcs + centers.at(0));
            for (int j = 1; j < centers.size(); j++)
            {
                double distance = adjacencyMatrix.at(i * numProcs + centers.at(j));
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }

            distancesSquared.push_back(minDistance * minDistance);
        }

        double total = std::accumulate(distancesSquared.begin(), distancesSquared.end(), 0.0);
        std::random_device totalRandom;
        std::mt19937 totalGen(totalRandom());
        std::uniform_real_distribution<double> totalDistribution(0, total);
        double threshold = totalDistribution(totalGen);
        double cumulative = 0;
        bool centerFound = false;
        for (int i = 0; i < numProcs && !centerFound; i++)
        {
            cumulative += distancesSquared.at(i);
            if (cumulative >= threshold)
            {
                centers.push_back(i);
                centerFound = true;
            }
        }
    }

    return centers;
}

bool updateCenterNodes(double* adjacencyMatrix, 
                       int numProcs, 
                       int* clusterAssignments, 
                       int* clusterSizes, 
                       int numClusters, 
                       int* clusterCenters)
{
    bool changed = false;
    for (int i = 0; i < numClusters; i++)
    {
        int clusterSize = clusterSizes[i];
        int clusterOffset = 0;
        double minEnergy = DBL_MAX;
        int minEnergyNode = -1;
        for (int j = 0; j < clusterSize; j++)
        {
            int possibleCenter = clusterAssignments[clusterOffset + j];
            double energy = 0.0;
            for (int k = 0; k < clusterSize; k++)
            {
                if (j != k)
                {
                    int node = clusterAssignments[clusterOffset + k];
                    double distance = adjacencyMatrix[node * numProcs + possibleCenter];
                    energy += distance * distance;
                }
            }

            if (energy < minEnergy)
            {
                minEnergy = energy;
                minEnergyNode = possibleCenter;
            }
        }

        if (minEnergyNode != clusterCenters[i])
        {
            clusterCenters[i] = minEnergyNode;
            changed = true;
        }
        clusterOffset += clusterSize;
    }
}

int main(int argc, char* argv[])
{
    int numNodes = 0;
    int numProcs = 0;
    int numSockets = 0;
    std::ifstream file;
    if (argc != 9)
    {
        std::cerr << "Invalid number of arguments. Expected 1, " << (argc - 1) << "received" << std::endl;
        return 1;
    }
    else
    {
        bool hasNodes = false;
        bool hasProcs = false;
        bool hasSockets = false;
        bool hasFilename = false;
        int i = 1;
        while (i < argc)
        {
            std::string flag = argv[i];
            i++;
            try {
                if (flag == "-n")
                {   
                    numNodes = std::stoi(std::string(argv[i]));
                    hasNodes = true;
                } 
                else if (flag == "-p")
                {
                    numProcs = std::stoi(std::string(argv[i]));
                    hasProcs = true;
                }
                else if (flag == "-s")
                {
                    numSockets = std::stoi(std::string(argv[i]));
                    hasSockets = true;
                }
                else if (flag == "-f")
                {
                    file = std::ifstream(std::string(argv[i]));
                    hasFilename = true;
                    if (!file.is_open())
                    {
                        std::cerr << "Error: could not open file." << std::endl;
                        return 1;
                    }
                } 
                else
                {
                    std::cerr << "Error: unknown argument " << flag << std::endl;
                }
            } 
            catch (const std::invalid_argument& e)
            {
                std::cerr << "Error: Invalid argument provided for flag " << flag << std::endl;
            }
        }

        if (!hasNodes || !hasProcs || !hasSockets || !hasFilename)
        {
            std::cerr << "Error: missing one or more required arguments." << std::endl;
            return 1;
        }
    }

    std::string line;
    std::vector<double> adjacencyMatrix;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string value;
        while (iss >> value)
        {
            adjacencyMatrix.push_back(std::stod(value));
        }
    }

    // k-means++ initilization
    std::vector<int> centers = initialize(numNodes, numSockets, numProcs);

    // rows are procs, cols are cluster slots
    int numCenters = numNodes * numSockets;
    int clusterSize = numProcs / numCenters;
    std::vector<int> clusterSizes(numCenters, clusterSize);
    bool changed;
    int* previousColAssignments = nullptr;
    do {
        double* costMatrix = calculateCostMatrix(centers.size(), centers.data(), clusterSizes.data(), numProcs, adjacencyMatrix.data());
        int* colAssignments = (int*) malloc(numProcs * sizeof(int));
        for (int i = 0; i < numProcs; i++)
        {
            colAssignments[i] = -1;
        }

        hungarian(costMatrix, numProcs, colAssignments);
        if (previousColAssignments == nullptr) 
        {
            previousColAssignments = colAssignments;
            changed = true;
        }
        else
        {
            for (int i = 0; i < numProcs; i++)
            {
                changed |= previousColAssignments[i] != colAssignments[i];
            }

            previousColAssignments = colAssignments;
        }

        changed != updateCenterNodes(adjacencyMatrix, numProcs, colAssignments, clusterSizes.data, numCenters, centers.data());

        free(colAssignments);
    } while (changed);

    printf("Cluster count: %d, Cluster Size: %d\n", numCenters, clusterSize);
    printf("Proc Assignments to cluster slots:\n");
    for (int i = 0; i < numProcs; i++)
    {
        printf("Cluster slot %d: %d\n", i, colAssignments[i]);
    }
}