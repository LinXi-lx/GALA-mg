#pragma once
#include <sys/time.h>
#include <nccl.h>
#include <mpi.h>
#include "graph/graph.h"
#include "graph/graph_gpu.h"

double louvain_gpu(Graph &g, vertex_t *h_comminity, double min_modularity,int gpu_id,int gpu_num);

double louvain_main_process(thrust::device_vector<weight_t> &d_weights,
                            thrust::device_vector<vertex_t> &d_neighbors,
                            thrust::device_vector<edge_t> &degrees,
                            thrust::device_vector<vertex_t> &coummunity_result,
                            thrust::device_vector<int> &primes,
                            vertex_t vertex_num, int &round,
                            double min_modularity, edge_t m2,int gpu_id,int gpu_num);

int build_compressed_graph(thrust::device_vector<weight_t> &d_weights,
                           thrust::device_vector<vertex_t> &d_neighbors,
                           thrust::device_vector<edge_t> &degrees,
                           thrust::device_vector<vertex_t>& coummunity_result,
                           thrust::device_vector<int> &primes, int vertex_num);

inline double get_time(){
    struct timeval time;
    gettimeofday(&time,NULL);
    return (double)time.tv_sec * 1000 + (double)time.tv_usec / 1000;
}