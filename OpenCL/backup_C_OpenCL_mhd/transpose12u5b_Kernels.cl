__kernel 
void 
transpose12u5b( 
	__global float4* t12_u1234,
	__global float* t12_u5,
	__global float4* t12_b,
	int t12_nx,
	int t12_ny,
	int t12_nz,
	__global float4* t12_u_update1234,
	__global float* t12_u_update5,
	__global float4* t12_b_update)
{     
        uint globalIdX = get_global_id(0);
        uint globalIdY = get_global_id(1);
        uint globalIdZ = get_global_id(2);

        uint groupIdX = get_group_id(0);
        uint groupIdY = get_group_id(1);
        uint groupIdZ = get_group_id(2);

        uint localIdX = get_local_id(0);
        uint localIdY = get_local_id(1);
        uint localIdZ = get_local_id(2);

	uint localSizeX = get_local_size(0);
	uint localSizeY = get_local_size(1);
	uint localSizeZ = get_local_size(2);

	uint numGroupsX = get_num_groups(0);
	uint numGroupsY = get_num_groups(1);
	uint numGroupsZ = get_num_groups(2);

/*
inside sub_cube:
localIdZ is k, threadIdx.z
localIdY is j, threadIdx.y
localIdX is i, threadIdx.x
*/
//	start
__local float block_u5[transpose_BLOCK_DIM][transpose_BLOCK_DIM][transpose_BLOCK_DIM+1];
__local float4 block_b[transpose_BLOCK_DIM][transpose_BLOCK_DIM][transpose_BLOCK_DIM+1];
//      read the data from global memory
unsigned int i_index =  groupIdX*transpose_BLOCK_DIM + localIdX;
unsigned int j_index =  groupIdY*transpose_BLOCK_DIM + localIdY;
unsigned int k_index =  groupIdZ*transpose_BLOCK_DIM + localIdZ;
////////////////////////////////
//
//      NOTES
//
//      NEED TO CHANGE IF MPI RUN ON MULTI-GPU
//
////////////////////////////////
if ((i_index<box_nx)&&(j_index<box_ny)&&(k_index<box_nz))
{
block_u5[localIdY][localIdX][localIdZ]=t12_u5[a3D_FinC((box_nx),(box_ny),(box_nz),i_index,j_index,k_index)];
block_b[localIdY][localIdX][localIdZ]=t12_b[a3D_FinC((box_nx),(box_ny),(box_nz),i_index,j_index,k_index)];
}
barrier(CLK_LOCAL_MEM_FENCE);
//      do the transpose
float4 temp_thread_b;
float temp_thread_b_float;
if ((i_index<box_nx)&&(j_index<box_ny)&&(k_index<box_nz))
{
//	get b
temp_thread_b=block_b[localIdX][localIdY][localIdZ];
//	transpose b
temp_thread_b_float=temp_thread_b.y;
temp_thread_b.y=temp_thread_b.x;
temp_thread_b.x=temp_thread_b_float;
//	put b
block_b[localIdX][localIdY][localIdZ]=temp_thread_b;
}
barrier(CLK_LOCAL_MEM_FENCE);
//
i_index =  groupIdX*transpose_BLOCK_DIM + localIdY;
j_index =  groupIdY*transpose_BLOCK_DIM + localIdX;
k_index =  groupIdZ*transpose_BLOCK_DIM + localIdZ;
//      write the data to global memory
if ((i_index<box_nx)&&(j_index<box_ny)&&(k_index<box_nz))
{
t12_u_update5[a3D_FinC((box_ny),(box_nx),(box_nz),j_index,i_index,k_index)]=block_u5[localIdX][localIdY][localIdZ];
t12_b_update[a3D_FinC((box_ny),(box_nx),(box_nz),j_index,i_index,k_index)]=block_b[localIdX][localIdY][localIdZ];
}
}
