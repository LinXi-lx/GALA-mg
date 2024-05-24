#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <nccl.h>
#include <mpi.h>
#include "kernel_functions.cuh"
#include "louvain.cuh"
using namespace std;



// #define CUDA_CHECK(cmd) do {                         
//   cudaError_t e = cmd;                              
//   if( e != cudaSuccess ) {                          
//     printf("Failed: Cuda error %s:%d '%s'\n",             
//         __FILE__,__LINE__,cudaGetErrorString(e));   
//     exit(EXIT_FAILURE);                             
//   }                                                 
// } while(0)



// #define NCCL_CHECK(cmd) do {                         
//   ncclResult_t r = cmd;                             
//   if (r!= ncclSuccess) {                            
//     printf("Failed, NCCL error %s:%d '%s'\n",             
//         __FILE__,__LINE__,ncclGetErrorString(r));   
//     exit(EXIT_FAILURE);                             
//   }                                                 
// }


int *read_primes(string file_name, int *prime_num)
{
	ifstream file(file_name);
	int *primes;
	if (file.is_open())
	{
		file >> *prime_num;
		int p;
		// std::cout << "Reading " << *prime_num << " prime numbers." <<
		// std::endl; Read primes in host memory
		int index = 0;
		primes = new int[*prime_num];
		while (file >> p)
		{

			primes[index++] = p;
			if (index >= *prime_num)
				break;
			// std::cout << aPrimeNum << " ";
		}
	}
	else
	{
		cout << "Can't open file containing prime numbers." << endl;
	}
	return primes;
}












// double louvain_phase_1(Graph &g, vertex_t *h_comminity,
// 				   double min_modularity,int round,int gpu_id,int gpu_num){
	
// 	thrust::device_vector<vertex_t> d_community(g.vertex_num);
// 	thrust::sequence(d_community.begin(), d_community.end());
// 	vertex_t community_num = g.vertex_num;
// 	GraphGPU g_gpu(g);
// 	edge_t m2 = 0; // total weight
// 	m2 = thrust::reduce(g_gpu.d_weights.begin(), g_gpu.d_weights.end(),
// 						(edge_t)0);

// 	thrust::device_vector<vertex_t> community_round(g.vertex_num);
// 	int prime_num;
// 	int *h_primes = read_primes("primes.txt", &prime_num);
// 	thrust::device_vector<int> primes(prime_num);
// 	thrust::copy(h_primes, h_primes + prime_num, primes.begin());

// 	// init end

	

	
// 	double start1, end1;
// 	start1 = get_time();
// 	louvain_main_process(g_gpu.d_weights, g_gpu.d_neighbors,
// 								   g_gpu.d_degrees, community_round, primes,
// 								   community_num, round, min_modularity, m2,gpu_id,gpu_num);
	

// 	thrust::gather(community_round.begin(), community_round.end(),
// 				   d_community.begin(), d_community.begin());
// 	ncclUniqueId id;
//   	ncclComm_t comm;
// 	cudaStream_t s;


// 	if (gpu_id == 0){
// 		ncclGetUniqueId(&id);
// 	}
// 	MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

// 	cudaStreamCreate(&s);
// 	int *sendbuff, *recvbuff;
// 	int size=5;
// 	cudaMalloc(&sendbuff, size * sizeof(int));
//   	cudaMalloc(&recvbuff, size * sizeof(int));
// 	int nums[5]={gpu_id,gpu_id+1,gpu_id+2,gpu_id+3,gpu_id+4};
// 	int recv[5]={0,0,0,0,0};
// 	cudaMemcpy(sendbuff,nums,size*sizeof(int),cudaMemcpyHostToDevice);
// 	cudaMemcpy(recvbuff,recv,size*sizeof(int),cudaMemcpyHostToDevice);
// 	ncclCommInitRank(&comm, gpu_num, id, gpu_id);
// 	ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclInt, /*ncclMax*/ ncclSum, comm,s);
	
// 	cudaStreamSynchronize(s);
// 	cudaMemcpy(nums,recvbuff,size*sizeof(int),cudaMemcpyDeviceToHost);
// 	printf("gpu-%d:%d %d %d %d %d\n",gpu_id,nums[0],nums[1],nums[2],nums[3],nums[4]);
// 	cudaFree(sendbuff);
//   	cudaFree(recvbuff);
// 	ncclCommDestroy(comm);

// 	return 0;
// }

double louvain_gpu(Graph &g, vertex_t *h_comminity,
				   double min_modularity,int gpu_id,int gpu_num)// return modularity
{ 	
	
	// cudaGetDeviceCount(&gpu_num);
	// thread threads[gpu_num];
	

	double prev_mod = -1, cur_mod = -1;
	

	int round = 0;
	
	cudaSetDevice(gpu_id);
	// louvain_phase_1(g, h_comminity,
	// 			   min_modularity, round,gpu_id,gpu_num);


	
	
	vertex_t community_num = g.vertex_num;

	//nccl
	

	thrust::device_vector<vertex_t> d_community(g.vertex_num);
	// trans to gpu
	GraphGPU g_gpu(g);

	edge_t m2 = 0; // total weight
	m2 = thrust::reduce(g_gpu.d_weights.begin(), g_gpu.d_weights.end(),
						(edge_t)0);

	thrust::device_vector<vertex_t> community_round(g.vertex_num);
	int prime_num;
	int *h_primes = read_primes("primes.txt", &prime_num);
	thrust::device_vector<int> primes(prime_num);
	thrust::copy(h_primes, h_primes + prime_num, primes.begin());

	// init end

	double start, end;
	start = get_time();

	printf("===============round:%d===============\n", round);
	double start1, end1;
	start1 = get_time();
	cur_mod = louvain_main_process(g_gpu.d_weights, g_gpu.d_neighbors,
								   g_gpu.d_degrees, community_round, primes,
								   community_num, round, min_modularity, m2,gpu_id, gpu_num);
	end1 = get_time();

	thrust::gather(community_round.begin(), community_round.end(),
				   d_community.begin(), d_community.begin());

	printf("louvain time in the first round = %fms\n", end1 - start1);

	// start1 = get_time();
	// community_num = build_compressed_graph(g_gpu.d_weights, g_gpu.d_neighbors,
	// 									   g_gpu.d_degrees, community_round,
	// 									   primes, community_num);
	// end1 = get_time();
	// printf("build time in the first round = %fms\n", end1 - start1);

	// printf("number of communities:%d modularity:%f\n", community_num, cur_mod);
	// // return 1;//!!!!!!!!!!!!!!!!!!
	// while (cur_mod - prev_mod >= min_modularity)
	// {

	// 	prev_mod = cur_mod;
	// 	printf("===============round:%d===============\n", round);
	// 	cur_mod = louvain_main_process(
	// 		g_gpu.d_weights, g_gpu.d_neighbors, g_gpu.d_degrees,
	// 		community_round, primes, community_num, round, min_modularity, m2);

	// 	thrust::gather(community_round.begin(), community_round.end(),
	// 				   d_community.begin(), d_community.begin());

	// 	community_num = build_compressed_graph(
	// 		g_gpu.d_weights, g_gpu.d_neighbors, g_gpu.d_degrees,
	// 		community_round, primes, community_num);

	// 	printf("number of communities:%d modularity:%f\n", community_num, cur_mod);
	// 	// print_CSR(&weights,&neighbors,&degrees,&community_num);
	// }
	// printf("=====================================\n");
	// end = get_time();
	// printf("final number of communities:%d --> %d final modularity:%f\n", g.vertex_num, community_num, cur_mod);
	// printf("execution time without data transfer = %fms\n", end - start);

	// thrust::copy(d_community.begin(), d_community.end(), h_comminity);

	return cur_mod;
}