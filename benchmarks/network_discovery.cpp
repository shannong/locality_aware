#include <mpi.h>

#include "mpi_advance.h"

#include <stdlib.h>
#include <stdio.h>
class PingPong
{
    public:

    PingPong(int _proc, MPI_Request* _request, int _even_odd)
    {
        proc = _proc;
        request = _request;
        tag = 0;
        even_odd = _even_odd;
        time = 0;
        sendbuf = 1;
        recvbuf = 0;
    }

    void start(int _n_iter)
    {
        n_iter = _n_iter;
        time = MPI_Wtime();
        step();
    }

    void ping()
    {
        MPI_Isend(&sendbuf, 1, MPI_FLOAT, proc, tag++, MPI_COMM_WORLD, request);
    }

    void pong()
    {
        MPI_Irecv(&recvbuf, 1, MPI_FLOAT, proc, tag++, MPI_COMM_WORLD, request);
    }

    int step()
    {
        if (tag / 2 == n_iter)
        {
            time = MPI_Wtime() - time;
            return 0;
        }

        if (tag % 2 == even_odd)
        {
            ping();
        }
        else
        {
            pong();
        }

        return 1;
    }


    int proc;
    int even_odd;
    float sendbuf;
    float recvbuf;
    MPI_Request* request;
    int tag;
    double time;
    int n_iter;
};

void dual_ping_pongs(PingPong** ping_pong, MPI_Request* req, int n_iter)
{
    // Start both ping pongs
    ping_pong[0]->start(n_iter);
    ping_pong[1]->start(n_iter);

    // Progress the ping pongs until n_iter iterations complete
    int active = 1;
    int idx;
    while (active)
    {
        // Wait for the current step of either ping pong to complete
        MPI_Waitany(2, req, &idx, MPI_STATUS_IGNORE);

        // Progress that ping pong
        active = ping_pong[idx]->step();
    }

    // Once a ping pong complete, progress only the other ping pong
    idx = (idx + 1) % 2;
    active = 1;
    while (active)
    {
        MPI_Wait(&(req[idx]), MPI_STATUS_IGNORE);
        active = ping_pong[idx]->step();
    }
}

double* naive_network_discovery(char* send_buffer, char* recv_buffer, int size, int tag, int num_iterations)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;

    double* times = (double*) malloc(num_procs * sizeof(double));
    times[rank] = 0.0;
    for (int i = 0; i < num_procs; i++)
    {
        for (int j = i + 1; j < num_procs; j++)
        {
            if (rank == i)
            {
                // warm up
                MPI_Send(send_buffer, size, MPI_CHAR, j, tag, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, size, MPI_CHAR, j, tag, MPI_COMM_WORLD, &status);

                double t0 = MPI_Wtime();
                for (int k = 0; k < num_iterations; k++)
                {
                    MPI_Send(send_buffer, size, MPI_CHAR, j, tag, MPI_COMM_WORLD);
                    MPI_Recv(recv_buffer, size, MPI_CHAR, j, tag, MPI_COMM_WORLD, &status);
                }

                times[j] = (MPI_Wtime() - t0) / (2. * num_iterations);
            }
            else if (rank == j)
            {
                // warm up
                MPI_Recv(recv_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
                MPI_Send(send_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD);

                double t0 = MPI_Wtime();
                for (int k = 0; k < num_iterations; k++)
                {
                    MPI_Recv(recv_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
                    MPI_Send(send_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD);
                }

                times[i] = (MPI_Wtime() - t0) / (2. * num_iterations);
            }
        }
    }

    return times;
}

double pingpong(float* buffer, int proc, int tag, int n_iter)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

int mod(int a, int m)
{
    return ((a % m) + m) % m;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    float buf;
    int proc, pos;
    int tag = 0;
    int n_iter = 1000;

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
            pos = 2 * r - 1;
            int j = mod(pos, n - 1);
            if (j == 0)
            {
                j = n - 1;
            }

            proc = j;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (proc < num_procs)
        {
            double t0 = pingpong(&buf, proc, tag, 1);
            double time = pingpong(&buf, proc, tag, n_iter);
            times[proc] = time;
        }

        tag++;
    }

    double adjacencyMatrix[num_procs * num_procs];
    MPI_Gather(times, num_procs, MPI_DOUBLE, adjacencyMatrix, num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < num_procs; i++)
        {
            for (int j = 0; j < num_procs; j++)
            {
                printf("%.10f\t", adjacencyMatrix[i * num_procs + j]);
            }

            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
