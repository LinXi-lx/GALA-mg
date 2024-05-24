#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include "graph/graph.h"
#include "louvain_gpu/louvain.cuh"
using namespace std;

int main(int argc, char **argv)
{   
    MPI_Init(&argc, &argv);//本行之后便进入多线程状态，线程数由 mpirun -np 4 ./a.out 的这个4来指定
    int gpu_num,gpu_id; 
    
  	MPI_Comm_rank(MPI_COMM_WORLD, &gpu_id);// 本线程的线程序号:myRank = 0, 1, 2, 4-1;
  	MPI_Comm_size(MPI_COMM_WORLD, &gpu_num);// 本次启动mpi 程序的总线程数 nRanks==4;
    cout<<gpu_num<<endl;
    string file_name;
    int is_weighted = 0;
    static const char *opt_string = "f:w";
    int opt = getopt(argc, argv, opt_string);
    while (opt != -1)
    {
        switch (opt)
        {
        case 'f':
            file_name = optarg;
            break;
        case 'w':
            is_weighted = true;
            break;
        }
        opt = getopt(argc, argv, opt_string);
    }

    double start = get_time();

    // cudaSetDevice(0);
    Graph g;
    g.load_bin_graph(file_name, is_weighted);

    cout << "load success" << endl;


    
    
    
    // cout<<gpu_id<<endl;
    vertex_t *community = (vertex_t *)malloc(sizeof(vertex_t) * (g.vertex_num));
    double curMod = louvain_gpu(g, community, 0.000001,gpu_id,gpu_num);


    MPI_Finalize();
    double end = get_time();

    printf("elapsed time = %fms\n", end - start);

    delete community;
}
