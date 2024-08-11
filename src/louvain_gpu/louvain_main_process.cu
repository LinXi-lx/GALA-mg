#include "kernel_functions.cuh"
#include "louvain.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
using namespace std;

void efficient_weight_updating(thrust::device_vector<weight_t> &d_weights,
							   thrust::device_vector<vertex_t> &d_neighbors,
							   thrust::device_vector<edge_t> &d_degrees,
							   thrust::device_vector<int> &cur_community,
							   thrust::device_vector<int> &sorted_vertex_id,
							   thrust::device_vector<uint8_t> &active_set,
							   thrust::device_vector<int> &In,
							   thrust::device_vector<int> &Tot,
							   thrust::device_vector<int> &K,
							   thrust::device_vector<int> &Self, int vertex_num,
							   int *deg_num_tbl, int min_Tot, double constant)
{
	int deg_num_4 = deg_num_tbl[0];
	int deg_num_8 = deg_num_tbl[1];
	int deg_num_16 = deg_num_tbl[2];
	int deg_num_32 = deg_num_tbl[3];
	int deg_num_128 = deg_num_tbl[4];
	int deg_num_1024 = deg_num_tbl[5];
	int deg_num_limit = deg_num_tbl[6];
	int deg_num_greater_than_limit = deg_num_tbl[7];
	int thread_num_to_alloc, block_num;
	if (deg_num_4)
	{
		thread_num_to_alloc = 4;
		block_num = (deg_num_4 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()),
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_4, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}

	if (deg_num_8)
	{
		thread_num_to_alloc = 8;

		block_num = (deg_num_8 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_8, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}
	if (deg_num_16)
	{
		thread_num_to_alloc = 16;
		block_num = (deg_num_16 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 + deg_num_8,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_16, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}

	if (deg_num_32)
	{
		thread_num_to_alloc = 32;
		block_num = (deg_num_32 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_32, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}

	if (deg_num_128)
	{ // 33-127
		thread_num_to_alloc = 32;

		block_num = (deg_num_128 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;
		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_128, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}

	if (deg_num_1024)
	{ // 128-1024
		// thread_num_to_alloc = 128;
		// int table_size = 257;
		int warp_size = 32;
		block_num = (deg_num_1024 * 256 + 256 - 1) / 256;

		compute_In_blk<<<block_num, 256>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_1024, warp_size,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}
	if (deg_num_limit)
	{ // 1025-4047

		int warp_size = 32;
		// int table_size=977;
		block_num = (deg_num_limit * 1024 + 1024 - 1) / 1024;

		compute_In_blk<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 + deg_num_1024,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_limit, warp_size,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}

	if (deg_num_greater_than_limit)
	{ // 4048-
		int warp_size = 32;
		// int table_size=6073;
		thread_num_to_alloc = 1024;
		block_num = (deg_num_greater_than_limit * thread_num_to_alloc + 1024 - 1) / 1024;

		// read primes

		compute_In_blk<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 + deg_num_1024 +
				deg_num_limit,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_greater_than_limit, warp_size,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()));
		cudaDeviceSynchronize();
	}
}

void filter_vertex_into_new_vector(thrust::device_vector<int> &sorted_vertex_id_const, thrust::device_vector<int> &sorted_vertex_id, vertex_filter<int> filter_, const int *deg_num_tbl_const, int *h_deg_num_tbl, int degree_type_size)
{
	thrust::device_vector<int>::iterator const_start = sorted_vertex_id_const.begin();
	thrust::device_vector<int>::iterator const_end = const_start;
	thrust::device_vector<int>::iterator filter_start = sorted_vertex_id.begin();
	thrust::device_vector<int>::iterator filter_end;
	for (int i = 0; i < degree_type_size; i++)
	{
		const_end = const_start + deg_num_tbl_const[i];
		filter_end = thrust::copy_if(const_start, const_end,
									 filter_start, filter_);
		h_deg_num_tbl[i] = thrust::distance(filter_start, filter_end);
		const_start = const_end;
		filter_start = filter_end;
	}
}

void filter_vertex_into_new_vector(thrust::device_vector<int> &sorted_vertex_id_const, thrust::device_vector<int> &sorted_vertex_id, vertex_filter<uint8_t> filter_, const int *deg_num_tbl_const, int *h_deg_num_tbl, int degree_type_size)
{
	thrust::device_vector<int>::iterator const_start = sorted_vertex_id_const.begin();
	thrust::device_vector<int>::iterator const_end = const_start;
	thrust::device_vector<int>::iterator filter_start = sorted_vertex_id.begin();
	thrust::device_vector<int>::iterator filter_end;
	for (int i = 0; i < degree_type_size; i++)
	{
		const_end = const_start + deg_num_tbl_const[i];
		filter_end = thrust::copy_if(const_start, const_end,
									 filter_start, filter_);
		h_deg_num_tbl[i] = thrust::distance(filter_start, filter_end);
		const_start = const_end;
		filter_start = filter_end;
	}
}

void get_deg_num_per_gpu(int* deg_num_tbl_const,int* deg_num_per_gpu,int gpu_id,int gpu_num,int size){
	int* temp=new int[size];
	temp[0]=+deg_num_tbl_const[0];
	for(int i=1;i<size;i++){

		temp[i]=temp[i-1]+deg_num_tbl_const[i];
	}
	for(int i=0;i<size;i++){

		deg_num_per_gpu[i]=temp[i]/gpu_num+(gpu_id<(temp[i]%gpu_num)?1:0);
	}
	for(int i=size-1;i>=1;i--){

		deg_num_per_gpu[i]=deg_num_per_gpu[i]-deg_num_per_gpu[i-1];
	}
	
}

__global__ void get_cur_community(int *cur_community, int *id_buffer, int *community_buffer, int buffer_size)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < buffer_size){
        int id=id_buffer[thread_id];
		if(id>=0){
			int comm=community_buffer[thread_id];
			cur_community[id]=comm;
		}
    }
}

// __global__ void prepare_stencil(int* stencil,int* sorted_)

double louvain_main_process(thrust::device_vector<weight_t> &d_weights,
							thrust::device_vector<vertex_t> &d_neighbors,
							thrust::device_vector<edge_t> &degrees,
							thrust::device_vector<vertex_t> &coummunity_result,
							thrust::device_vector<int> &primes,
							vertex_t vertex_num, int &round,
							double min_modularity, edge_t m2,int gpu_id,int gpu_num)
{
	// init gpu vector
	int iteration = 0;
	// round=0;
	ncclUniqueId id;
  	ncclComm_t comm;
	cudaStream_t s;


	if (gpu_id == 0){
		ncclGetUniqueId(&id);
	}
	MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

	cudaStreamCreate(&s);
	// int *sendbuff;
	// cudaMalloc(&sendbuff, 8*vertex_num * sizeof(int));
  	// cudaMalloc(&recvbuff, vertex_num * sizeof(int));


	thrust::device_vector<edge_t> d_degrees(vertex_num + 1);
	thrust::copy(thrust::device, degrees.begin(), degrees.end(),
				 d_degrees.begin() + 1); // d_degrees[0]=0

	thrust::device_vector<int> degree_of_vertex(vertex_num, 0);
	thrust::transform(d_degrees.begin() + 1, d_degrees.end(), d_degrees.begin(),
					  degree_of_vertex.begin(),
					  thrust::minus<edge_t>()); // d[i]-d[i-1]

	double constant = 1 / (double)m2;
	// init filter

	int global_limit = SHARE_MEM_SIZE * 2 / 3 - 1; // 2703,vertex degree under limit can use share mem//!!!!!!!!!!!!!!!

	degree_filter_in_range filter_for_4(1, 3);
	degree_filter_in_range filter_for_8(4, 7);
	degree_filter_in_range filter_for_16(8, 15);
	degree_filter_in_range filter_for_32(16, 31);
	degree_filter_in_range filter_for_128(32, 128);
	degree_filter_in_range filter_for_1024(129, 1024);
	degree_filter_in_range filter_for_limit(1025, global_limit);
	degree_filter_in_range filter_for_global(global_limit + 1, vertex_num);

	int h_degree_type_low[] = {1, 4, 8, 16, 32, 129, 1025, global_limit + 1};
	int h_degree_type_high[] = {3, 7, 15, 31, 128, 1024, global_limit, vertex_num};
	int degree_type_size = 8;
	thrust::device_vector<int> degree_type_low(degree_type_size);
	thrust::device_vector<int> degree_type_high(degree_type_size);
	thrust::copy(h_degree_type_low, h_degree_type_low + degree_type_size,
				 degree_type_low.begin());
	thrust::copy(h_degree_type_high, h_degree_type_high + degree_type_size,
				 degree_type_high.begin());

	// count degree
	int deg_num_4 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									 degree_of_vertex.end(), filter_for_4);
	int deg_num_8 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									 degree_of_vertex.end(), filter_for_8);
	int deg_num_16 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									  degree_of_vertex.end(), filter_for_16);
	int deg_num_32 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									  degree_of_vertex.end(), filter_for_32);
	int deg_num_128 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									   degree_of_vertex.end(), filter_for_128);
	int deg_num_1024 =
		thrust::count_if(thrust::device, degree_of_vertex.begin(),
						 degree_of_vertex.end(), filter_for_1024);
	int deg_num_limit =
		thrust::count_if(thrust::device, degree_of_vertex.begin(),
						 degree_of_vertex.end(), filter_for_limit);
	int deg_num_greater_than_limit =
		thrust::count_if(thrust::device, degree_of_vertex.begin(),
						 degree_of_vertex.end(), filter_for_global);

	int deg_num_tbl_const[] = {
		deg_num_4, deg_num_8, deg_num_16, deg_num_32,
		deg_num_128, deg_num_1024, deg_num_limit, deg_num_greater_than_limit};

	// init Tot, In...
	thrust::device_vector<int> Tot(vertex_num, 0); // comunity-tot degree (N: 0-maxcommunity)
	thrust::device_vector<int> In(vertex_num, 0);  // comunity-in degree (N: 0-maxcommunity)//don't need modify?
	thrust::device_vector<int> next_In(vertex_num, 0);
	thrust::device_vector<int> K(vertex_num, 0); // degree of merge-vertex i
	// thrust::device_vector<int> community(vertex_num);//community of
	// merge-vertex i thrust::sequence(community.begin(),community.end());
	thrust::device_vector<int> Self(vertex_num, 0);
	// community labels

	thrust::device_vector<int> prev_community(vertex_num);
	thrust::device_vector<int> cur_community(vertex_num);

	thrust::device_vector<int> Tot_update(vertex_num,
										  0); // record update information
	thrust::device_vector<int> community_size(vertex_num, 1);
	thrust::device_vector<int> community_size_update(vertex_num, 0);
	int community_num = vertex_num; // number of community

	// fill Tot,K,Self
	int warp_size = WARP_SIZE;
	int block_num = (community_num + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK; // 1 thread -> 1 vertex
	init_communities<<<block_num, THREAD_NUM_PER_BLOCK>>>(
		thrust::raw_pointer_cast(d_weights.data()),
		thrust::raw_pointer_cast(d_neighbors.data()),
		thrust::raw_pointer_cast(d_degrees.data()),
		thrust::raw_pointer_cast(K.data()),
		thrust::raw_pointer_cast(Tot.data()),
		thrust::raw_pointer_cast(Self.data()),
		thrust::raw_pointer_cast(prev_community.data()), vertex_num, warp_size);
	cur_community = prev_community;

	// init sorted list for bucket
	int deg_num_4_per_gpu, deg_num_8_per_gpu, deg_num_16_per_gpu, deg_num_32_per_gpu,
		deg_num_128_per_gpu, deg_num_1024_per_gpu, deg_num_limit_per_gpu, deg_num_greater_than_limit_per_gpu;
	// int* data_num_per_gpu_list_offset=new int[gpu_num+1];
	// data_num_per_gpu_list_offset[0]=0;
	// for(int i=1;i<gpu_num+1;i++){
	// 	data_num_per_gpu_list_offset[i]=data_num_per_gpu_list_offset[i-1]+vertex_num/gpu_num+(i<(vertex_num%gpu_num)?1:0);
	// }
	// int data_num_per_gpu=data_num_per_gpu_list_offset[gpu_id+1]-data_num_per_gpu_list_offset[gpu_id];
	int data_num_per_gpu=vertex_num/gpu_num+(gpu_id<(vertex_num%gpu_num)?1:0);

	thrust::device_vector<int> sorted_vertex_id_all(vertex_num);
	// thrust::device_vector<int> sorted_vertex_id_gpu_distribute(vertex_num);
	thrust::device_vector<int> sorted_vertex_id(data_num_per_gpu);
	thrust::device_vector<int> degree_of_vertex_per_gpu(data_num_per_gpu);

	

	thrust::sequence(sorted_vertex_id_all.begin(), sorted_vertex_id_all.end());
	thrust::sort_by_key(degree_of_vertex.begin(),degree_of_vertex.end(),
		sorted_vertex_id_all.begin() );
	// thrust::sort_by_key(
	// 	degree_of_vertex.begin() + deg_num_4 + deg_num_8 + deg_num_16 + deg_num_32 +
	// 		deg_num_128 + deg_num_1024 + deg_num_limit,
	// 	original_vertex_id.end(),
	// 	sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 + deg_num_32 +
	// 		deg_num_128 + deg_num_1024 + deg_num_limit,
	// 	thrust::greater<unsigned int>());
	
	// thrust::device_vector<int> d_indices(vertex_num);
    // thrust::sequence(d_indices.begin(), d_indices.end());
	// thrust::counting_iterator<int> idxfirst(0);
  	// thrust::counting_iterator<int> idxlast = idxfirst +vertex_num;

    thrust::device_vector<int> index(vertex_num);
	thrust::sequence(index.begin(), index.end());

	thrust::copy_if(sorted_vertex_id_all.begin(),sorted_vertex_id_all.end(),index.begin(),sorted_vertex_id.begin(),index_gpu_filter(gpu_id,gpu_num));
    // for(int i=0;i<gpu_num;i++){
	// 	thrust::copy_if(sorted_vertex_id_all.begin(),sorted_vertex_id_all.end(),index.begin(),sorted_vertex_id_gpu_distribute.begin()+data_num_per_gpu_list_offset[i],index_gpu_filter(i,gpu_num));
	// }

	thrust::copy_if(degree_of_vertex.begin(),degree_of_vertex.end(),index.begin(),degree_of_vertex_per_gpu.begin(),index_gpu_filter(gpu_id,gpu_num));
	
	int* deg_num_per_gpu=new int[degree_type_size];
	get_deg_num_per_gpu(deg_num_tbl_const,deg_num_per_gpu,gpu_id, gpu_num, degree_type_size);
	deg_num_4_per_gpu = deg_num_per_gpu[0];
	deg_num_8_per_gpu = deg_num_per_gpu[1];
	deg_num_16_per_gpu = deg_num_per_gpu[2];
	deg_num_32_per_gpu = deg_num_per_gpu[3];
	deg_num_128_per_gpu = deg_num_per_gpu[4];
	deg_num_1024_per_gpu = deg_num_per_gpu[5];
	deg_num_limit_per_gpu= deg_num_per_gpu[6];
	deg_num_greater_than_limit_per_gpu = deg_num_per_gpu[7];

	cout<<"gpu-"<<gpu_id<<":"<<deg_num_4_per_gpu<<" "<<deg_num_8_per_gpu<<" "<<deg_num_16_per_gpu
	<<" "<<deg_num_32_per_gpu<<" "<<deg_num_128_per_gpu<<" "<<deg_num_1024_per_gpu<<" "
	<<deg_num_limit_per_gpu<<" "<<deg_num_greater_than_limit_per_gpu<<endl;


	thrust::device_vector<int> stencil(vertex_num,1);
	thrust::device_vector<int> temp(data_num_per_gpu,0);
	thrust::scatter(temp.begin(),temp.end(),sorted_vertex_id.begin(),stencil.begin());//1:other gpu



	
	// prepare global table start location for large vertices
	thrust::device_vector<int> global_table_offset(deg_num_greater_than_limit_per_gpu + 1, 0);

	if (deg_num_greater_than_limit_per_gpu > 0)
	{
		block_num = (deg_num_greater_than_limit_per_gpu * WARP_SIZE + 1024 - 1) / 1024;
		compute_global_table_size_louvain<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(primes.data()), primes.size(),
			thrust::raw_pointer_cast(degree_of_vertex_per_gpu.data()) +deg_num_4_per_gpu+deg_num_8_per_gpu
				+deg_num_16_per_gpu+deg_num_32_per_gpu+deg_num_128_per_gpu
				+deg_num_1024_per_gpu+deg_num_limit_per_gpu,
			deg_num_greater_than_limit_per_gpu, thrust::raw_pointer_cast(global_table_offset.data()) + 1,
			warp_size);
	}

	thrust::inclusive_scan(global_table_offset.begin(), global_table_offset.end(),
						   global_table_offset.begin(), thrust::plus<int>());
	thrust::device_vector<int> global_table(2 * global_table_offset.back(), 0);

	// thrust::device_vector<int> global_table(4 * global_table_offset.back(), 0); // global
	// hash table return 1;

	thrust::device_vector<uint8_t> active_set(vertex_num, 1); // active set
	thrust::device_vector<int> is_moved(vertex_num, 0);	  // whether vertex has moved into new comm in the iteration

	thrust::device_vector<int> sorted_vertex_id_const(
		data_num_per_gpu); // save a sorted_vertex_id copy
	thrust::copy(thrust::device, sorted_vertex_id.begin(), sorted_vertex_id.end(),
				 sorted_vertex_id_const.begin());


	double prev_modularity = -1;
	double cur_modularity = -1;
	int thread_num_to_alloc;

	thrust::device_vector<int> target_com_weights(vertex_num, 0);
	
	ncclCommInitRank(&comm, gpu_num, id, gpu_id);
	// int* In_sums;
    // cudaMalloc(&In_sums, sizeof(int)*gpu_num);
	thrust::device_vector<int> In_sums(gpu_num);
	double start, end;
	start = get_time();

	efficient_weight_updating(
		d_weights, d_neighbors, d_degrees, cur_community, sorted_vertex_id_all,
		active_set, In, Tot, K, Self, vertex_num, deg_num_tbl_const, 1, constant);
	thrust::device_vector<double> sum(vertex_num, 0);
	thrust::transform(thrust::device, In.begin(), In.end(), Tot.begin(),
					  sum.begin(), modularity_op(constant));
	cur_modularity = thrust::reduce(thrust::device, sum.begin(), sum.end(),
									(double)0.0, thrust::plus<double>());
	
	if(gpu_id==0)
			printf("gpu-%d: Iteration:%d Q:%f\n",gpu_id, iteration, cur_modularity);

	iteration++;
	prev_modularity = cur_modularity;

	// save vertices whose neighbors do not moving in the iteration

	double decideandmovetime = 0;
	double updatetime = 0;
	double remainingtime = 0;
	double comm_time=0;
	double test_time=0;
	// cout<<gpu_id<<"sorted_vertex_id:";
	// for(int i=0;i<deg_num_4_per_gpu+deg_num_8_per_gpu+deg_num_16_per_gpu+deg_num_32_per_gpu+deg_num_128_per_gpu+deg_num_1024_per_gpu+deg_num_limit_per_gpu+deg_num_greater_than_limit_per_gpu;i++){
	// 	cout<<sorted_vertex_id[i]<<" ";
	// }
	// cout<<endl;
	// cout<<gpu_id<<"stencil:";
	// for(int i=0;i<vertex_num;i++){
	// 	cout<<stencil[i]<<" ";
	// }
	// cout<<endl;
	// thrust::device_vector<int> sendbuffer(gpu_num*data_num_per_gpu_list_offset[1]);
	while (true)
	{
		
		thrust::fill(active_set.begin(), active_set.end(), 0);
		thrust::fill(is_moved.begin(), is_moved.end(), 0);

		thrust::fill(target_com_weights.begin(), target_com_weights.end(), 0);

		// thrust::replace_if(cur_community.begin(),cur_community.end(),stencil.begin(),[] __device__ (int x) { return x  == 1; },0);

		

		double start1, end1;
		// double time[8];
		start1 = get_time();

		if (deg_num_4_per_gpu)
		{
			thread_num_to_alloc = 4;

			block_num = (deg_num_4_per_gpu * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()),
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_4_per_gpu,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		// end1=get_time();
		// time[0]=end1-start1;
		// start1=get_time();
		if (deg_num_8_per_gpu)
		{
			thread_num_to_alloc = 8;

			block_num = (deg_num_8_per_gpu * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_8_per_gpu,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);

			cudaDeviceSynchronize();
		}
		// end1=get_time();
		// time[1]=end1-start1;
		// start1=get_time();
		if (deg_num_16_per_gpu)
		{
			thread_num_to_alloc = 16;

			block_num = (deg_num_16_per_gpu * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu +
					deg_num_8_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_16_per_gpu,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		// end1=get_time();
		// time[2]=end1-start1;
		// start1=get_time();
		if (deg_num_32_per_gpu)
		{
			thread_num_to_alloc = 32;

			block_num = (deg_num_32_per_gpu * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu +
					deg_num_8_per_gpu + deg_num_16_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_32_per_gpu,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		// end1=get_time();
		// time[3]=end1-start1;
		// start1=get_time();
		if (deg_num_128_per_gpu)
		{ // 33-127
			thread_num_to_alloc = 32;
			int table_size = 257;
			block_num = (deg_num_128_per_gpu * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;
			int shared_size =
				table_size * (THREAD_NUM_PER_BLOCK / thread_num_to_alloc);
			decide_and_move_hash_shared<<<block_num, THREAD_NUM_PER_BLOCK,
										  3 * shared_size * sizeof(int)>>>(
				// find_best_com_blk_no_share<<<deg_num_128, 128,2 * table_size *
				// sizeof(int)>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu +
					deg_num_8_per_gpu + deg_num_16_per_gpu + deg_num_32_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_128_per_gpu,
				table_size, thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		// end1=get_time();
		// time[4]=end1-start1;
		// start1=get_time();
		if (deg_num_1024_per_gpu)
		{ // 129-1024
			warp_size = 32;
			block_num = (deg_num_1024_per_gpu * 256 + 256 - 1) / 256;
			decide_and_move_hash_hierarchical<<<block_num, 256>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu +
					deg_num_8_per_gpu + deg_num_16_per_gpu + deg_num_32_per_gpu + deg_num_128_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()),
				thrust::raw_pointer_cast(global_table_offset.data()),
				thrust::raw_pointer_cast(global_table.data()),
				thrust::raw_pointer_cast(primes.data()), primes.size(),
				deg_num_1024_per_gpu, warp_size, global_limit, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		// end1=get_time();
		// time[5]=end1-start1;
		// start1=get_time();
		if (deg_num_limit_per_gpu)
		{ // 1025-2703
			warp_size = 32;
			block_num = (deg_num_limit_per_gpu * MAX_THREAD_PER_BLOCK +
						 MAX_THREAD_PER_BLOCK - 1) /
						MAX_THREAD_PER_BLOCK;
			decide_and_move_hash_hierarchical<<<block_num, MAX_THREAD_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu +
					deg_num_8_per_gpu + deg_num_16_per_gpu + deg_num_32_per_gpu + deg_num_128_per_gpu +
					deg_num_1024_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()),
				thrust::raw_pointer_cast(global_table_offset.data()),
				thrust::raw_pointer_cast(global_table.data()),
				thrust::raw_pointer_cast(primes.data()), primes.size(),
				deg_num_limit_per_gpu, warp_size, global_limit, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		// end1=get_time();
		// time[6]=end1-start1;
		// start1=get_time();
		if (deg_num_greater_than_limit_per_gpu)
		{ // 2704-
			warp_size = 32;
			block_num = (deg_num_greater_than_limit_per_gpu * MAX_THREAD_PER_BLOCK +
						 MAX_THREAD_PER_BLOCK - 1) /
						MAX_THREAD_PER_BLOCK;
			// read primes
			decide_and_move_hash_hierarchical<<<block_num, 1024>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4_per_gpu +
					deg_num_8_per_gpu + deg_num_16_per_gpu + deg_num_32_per_gpu + deg_num_128_per_gpu +
					deg_num_1024_per_gpu + deg_num_limit_per_gpu,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()),
				thrust::raw_pointer_cast(global_table_offset.data()),
				thrust::raw_pointer_cast(global_table.data()),
				thrust::raw_pointer_cast(primes.data()), primes.size(),
				deg_num_greater_than_limit_per_gpu, warp_size, global_limit, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		end1 = get_time();
	

		decideandmovetime += end1 - start1;
		
		
		double start2, end2;
		start2 = get_time();
	

		double start_comm, end_comm;
		start_comm = get_time();

		thrust::device_vector<int> id_buffer(vertex_num, -1);
		thrust::sequence(sorted_vertex_id_all.begin(), sorted_vertex_id_all.end());
		thrust::device_vector<int> communitybuffer(vertex_num, 0);
		thrust::device_vector<int>::iterator buffer_end = thrust::copy_if(cur_community.begin(), cur_community.end(), is_moved.begin(),
																		communitybuffer.begin(), [] __device__ (int x) { return x  > 0; });
		buffer_end = thrust::copy_if(sorted_vertex_id_all.begin(), sorted_vertex_id_all.end(), is_moved.begin(),
		                            id_buffer.begin(), [] __device__ (int x) { return x  > 0; });
		
		int id_size = thrust::distance(id_buffer.begin(), buffer_end);//非0个数
		int id_max_size;

		// for(int i=0;i<id_size;i++){
		// 	cout<<id_buffer[i]<<" ";
		// }cout<<endl;
		MPI_Allreduce(&id_size, &id_max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		// cout<<id_max_size<<endl;
		thrust::device_vector<int> recvbuffer_id(id_max_size*gpu_num,-1);
		thrust::device_vector<int> recvbuffer_comm(id_max_size*gpu_num);
		ncclAllGather((const void*)thrust::raw_pointer_cast(id_buffer.data()), (void*)thrust::raw_pointer_cast(recvbuffer_id.data()), id_max_size, ncclInt, comm,s);
		ncclAllGather((const void*)thrust::raw_pointer_cast(communitybuffer.data()), (void*)thrust::raw_pointer_cast(recvbuffer_comm.data()), id_max_size, ncclInt, comm,s);
		

		// ncclAllReduce((const void*)thrust::raw_pointer_cast(cur_community.data()), (void*)thrust::raw_pointer_cast(cur_community.data()), vertex_num, ncclInt, /*ncclMax*/ ncclSum, comm,s);
		ncclAllReduce((const void*)thrust::raw_pointer_cast(active_set.data()), (void*)thrust::raw_pointer_cast(active_set.data()), vertex_num, ncclUint8, /*ncclMax*/ ncclSum, comm,s);//标记邻居动的情况

		ncclAllReduce((const void*)thrust::raw_pointer_cast(target_com_weights.data()), (void*)thrust::raw_pointer_cast(target_com_weights.data()), vertex_num, ncclInt, /*ncclMax*/ ncclSum, comm,s);//邻居移动对点产生的delta

		
		cudaStreamSynchronize(s);
		// cout<<"buffer: ";
		// for(int i=0;i<id_max_size*gpu_num;i++){
		// 	cout<<recvbuffer_id[i]<<" ";
		// }cout<<endl;
		// cout<<"comm buffer: ";
		// for(int i=0;i<id_max_size*gpu_num;i++){
		// 	cout<<recvbuffer_comm[i]<<" ";
		// }cout<<endl;
		block_num = (id_max_size*gpu_num + 1024 - 1) / 1024;
		get_cur_community<<<block_num, 1024>>>(thrust::raw_pointer_cast(cur_community.data()), thrust::raw_pointer_cast(recvbuffer_id.data()), 
		thrust::raw_pointer_cast(recvbuffer_comm.data()), id_max_size*gpu_num);

		end_comm=get_time();
		comm_time+=end_comm-start_comm;
		
		
		// compute In

		thrust::fill(Tot.begin(), Tot.end(),0);
		thrust::fill(community_size.begin(), community_size.end(),0);
		block_num = (vertex_num + 1024 - 1) / 1024;	
		get_Tot_and_comm_size<<<block_num, 1024>>>(thrust::raw_pointer_cast(Tot.data()), thrust::raw_pointer_cast(community_size.data()), 
		thrust::raw_pointer_cast(cur_community.data()), thrust::raw_pointer_cast(K.data()), vertex_num);
		edge_t min_Tot = thrust::transform_reduce(Tot.begin(), Tot.end(), Tot_op(m2), m2,
												  thrust::minimum<edge_t>());

		// min_Tot=1;
		
		block_num = (vertex_num + 1024 - 1) / 1024;									
		save_next_In<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(next_In.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(active_set.data()),
			thrust::raw_pointer_cast(is_moved.data()),
			thrust::raw_pointer_cast(target_com_weights.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(Self.data()),
			thrust::raw_pointer_cast(K.data()),thrust::raw_pointer_cast(stencil.data()), (int)min_Tot, constant,
			vertex_num, iteration);
		cudaDeviceSynchronize();
		
		int h_deg_num_tbl[degree_type_size];
		vertex_filter<int> moved_filter(is_moved.data(), 1); // vertices to compute In , is_moved 标记邻居动且自身动的情况
		thrust::device_vector<int> compute_In_sorted_vertex_id(vertex_num);
		filter_vertex_into_new_vector(sorted_vertex_id_const, compute_In_sorted_vertex_id, moved_filter, deg_num_per_gpu, h_deg_num_tbl, degree_type_size);

		efficient_weight_updating(d_weights, d_neighbors, d_degrees,
								  cur_community, compute_In_sorted_vertex_id,
								  active_set, In, Tot, K, Self, vertex_num,
								  h_deg_num_tbl, (int)min_Tot, constant);
		end2 = get_time();
		updatetime += end2 - start2;

		// ncclAllReduce((const void*)thrust::raw_pointer_cast(In.data()), (void*)thrust::raw_pointer_cast(In.data()), vertex_num, ncclInt, /*ncclMax*/ ncclSum, comm,s);
		// cudaStreamSynchronize(s);
		// for(int i=0;i<vertex_num;i++){
		// 	cout<<In[i]<<" ";
		// }cout<<endl;

		// thrust::replace_if(In.begin(),In.end(),stencil.begin(),[] __device__ (int x) { return x  == 1; },0);

		int In_sum=thrust::reduce(thrust::device, In.begin(), In.end(),
										0, thrust::plus<int>());
		// double start_test=get_time();
		MPI_Allreduce(&In_sum, &In_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		// double end_test=get_time();
		// test_time+=end_test-start_test;
		thrust::device_vector<double> sum(vertex_num, 0);
		thrust::transform(thrust::device, Tot.begin(), Tot.end(), 
						  sum.begin(), modularity_op_(constant));
		cur_modularity = In_sum*constant+thrust::reduce(thrust::device, sum.begin(), sum.end(),
										(double)0.0, thrust::plus<double>());
		

		if(gpu_id==0)
			printf("gpu-%d: Iteration:%d Q:%f\n",gpu_id, iteration, cur_modularity);
		if ((cur_modularity - prev_modularity) < min_modularity)
		{ // threshold
			break;
		}
		
		vertex_filter<uint8_t> active_filter(active_set.data(), 1);
		filter_vertex_into_new_vector(sorted_vertex_id_const, sorted_vertex_id, active_filter, deg_num_per_gpu, h_deg_num_tbl, degree_type_size);
		deg_num_4_per_gpu = h_deg_num_tbl[0];
		deg_num_8_per_gpu = h_deg_num_tbl[1];
		deg_num_16_per_gpu = h_deg_num_tbl[2];
		deg_num_32_per_gpu = h_deg_num_tbl[3];
		deg_num_128_per_gpu = h_deg_num_tbl[4];
		deg_num_1024_per_gpu = h_deg_num_tbl[5];
		deg_num_limit_per_gpu = h_deg_num_tbl[6];
		deg_num_greater_than_limit_per_gpu = h_deg_num_tbl[7];
		
		// cout<<"sorted_vertex_id:";
		// for(int i=0;i<deg_num_4_per_gpu+deg_num_8_per_gpu+deg_num_16_per_gpu+deg_num_32_per_gpu+deg_num_128_per_gpu+deg_num_1024_per_gpu+deg_num_limit_per_gpu+deg_num_greater_than_limit_per_gpu;i++){
		// 	cout<<sorted_vertex_id[i]<<" ";
		// }
		// cout<<endl;
		
		prev_modularity = cur_modularity;

		// thrust::device_ptr<int> temp=prev_community.begin();
		prev_community = cur_community; // Current holds the chosen assignment
		// thrust::swap(prev_community,cur_community);
		iteration++;
		
		// return 1;
	}

	round++;

	end = get_time();

	printf("gpu-%d: time without data init = %fms\n",gpu_id, end - start);

	remainingtime = end - start - decideandmovetime - updatetime;
	printf("gpu-%d: decideandmove time = %fms weight updating time = %fms remaining time = %fms\n",gpu_id, decideandmovetime, updatetime, remainingtime);
	printf("gpu-%d: comm time = %fms \n",gpu_id, comm_time);
	printf("gpu-%d: test time = %fms \n",gpu_id, test_time);
	// printf("gpu-%d: Iteration:%d Q:%f\n",gpu_id, iteration, prev_modularity);
	coummunity_result.resize(vertex_num);
	thrust::copy(prev_community.begin(), prev_community.end(), coummunity_result.begin());
	// prev_community.clear();
	// cur_community.clear();
	// Tot.clear();
	// In.clear();
	// next_In.clear();
	// K.clear();
	// Self.clear();
	// community_size.clear();
	// community_size_update.clear();
	// Tot_update.clear();
	return prev_modularity;
}
