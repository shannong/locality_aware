#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

typedef struct {
    int node;
    double distance;
} NodeDistance;

double mahalanobis_distance(std::vector<NodeDistance> cluster, NodeDistance point)
{
    double mean = std::accumulate(cluster.begin(), cluster.end(), 0.0,
                                  [](double sum, const NodeDistance& nd) {
                                        return sum + nd.distance;
                                  });
    mean /= cluster.size();

    double std_dev = std::accumulate(cluster.begin(), cluster.end(), 0.0,
                                     [mean](double sum, const NodeDistance& nd) {
                                        double difference = nd.distance - mean;
                                        difference *= difference;
                                        return sum + difference;
                                     });
    std_dev /= cluster.size();
    std_dev = std::sqrt(std_dev);

    return std::abs(point.distance - mean) / std_dev;
}

int main(int argc, char** argv)
{
    int num_nodes = 2;
    int count = 128;
    // std::ifstream dataFile("/Users/shannon/dev/locality_aware/test_data/test_networkdiscovery-100000.out");
    std::ifstream dataFile("/Users/shannon/dev/locality_aware/test_data/networkDiscover_rank0Naive.out");
    std::string line;
    std::vector<NodeDistance> adjacencyMatrixData{};
    double value;
    std::getline(dataFile, line);
    std::stringstream ss(line);
    int node = 0;
    while(ss >> value)        
    {
        NodeDistance nodeDistance = {node, value};
        adjacencyMatrixData.push_back(nodeDistance);
        node++;   
    }

    std::sort(adjacencyMatrixData.begin(), adjacencyMatrixData.end(), 
          [](const NodeDistance& a, const NodeDistance& b) {
              return a.distance < b.distance;
          });

    std::ofstream csvFile("rank0_cluster_naive_md.csv");
    if (!csvFile.is_open()) {
        std::fprintf(stderr, "Failed to open rank0_cluster_md.csv for writing\n");
        return 1;
    }
    csvFile << "cluster size,cluster md,other md,ratio\n";

    printf("Adjacency matrix size: %zu\n", adjacencyMatrixData.size());
    int maxClusterSize = count / num_nodes;
    // 0 and 1 node communicators don't make sense.
    for (int i = 2; i <= maxClusterSize; i++)
    {
        // if (maxClusterSize % (i + 1) == 0)
        // {
            // communicators need to be equal sized
            std::vector<NodeDistance> local_cluster(adjacencyMatrixData.begin(), adjacencyMatrixData.begin() + i);
            std::vector<NodeDistance> everything_else(adjacencyMatrixData.begin() + i + 1, adjacencyMatrixData.end());

            double cluster_md = mahalanobis_distance(local_cluster, adjacencyMatrixData.at(i + 1));
            double other_md = mahalanobis_distance(everything_else, adjacencyMatrixData.at(i + 1));
            double ratio = cluster_md / other_md;
            printf("i = %d, Cluster size: %zu, Cluster MD: %f, Other Size: %zu, Other MD: %f, Ratio: %f\n", i, local_cluster.size(), cluster_md, everything_else.size(), other_md, ratio);
            csvFile << local_cluster.size() << ',' << cluster_md << ',' << other_md << ',' << ratio << '\n';
        // }
    }

    for (int i = 0; i < adjacencyMatrixData.size(); i++)
    {
        printf("%d: %d (%f)\n", i, adjacencyMatrixData.at(i).node, adjacencyMatrixData.at(i).distance);
    }
}
