__kernel 
void 
transpose13u1234( 
	__global float4* t13_u1234,
	__global float* t13_u5,
	__global float4* t13_b,
	int t13_nx,
	int t13_ny,
	int t13_nz,
	__global float4* t13_u_update1234,
	__global float* t13_u_update5,
	__global float4* t13_b_update)
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
//      start
__local float4 block_u1234[transpose_BLOCK_DIM][transpose_BLOCK_DIM][transpose_BLOCK_DIM+1];
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
block_u1234[localIdZ][localIdY][localIdX]=t13_u1234[a3D_FinC((box_nx),(box_ny),(box_nz),i_index,j_index,k_index)];
}
barrier(CLK_LOCAL_MEM_FENCE);
//      do the transpose
float4 temp_thread_u1234;
float temp_thread_u_float;
if ((i_index<box_nx)&&(j_index<box_ny)&&(k_index<box_nz))
{
//      get u 
temp_thread_u1234=block_u1234[localIdX][localIdY][localIdZ];
//      transpose u
temp_thread_u_float=temp_thread_u1234.y;
temp_thread_u1234.y=temp_thread_u1234.w;
temp_thread_u1234.w=temp_thread_u_float;
//      put u 
block_u1234[localIdX][localIdY][localIdZ]=temp_thread_u1234;
}
barrier(CLK_LOCAL_MEM_FENCE);
//
i_index =  groupIdX*transpose_BLOCK_DIM + localIdZ;
j_index =  groupIdY*transpose_BLOCK_DIM + localIdY;
k_index =  groupIdZ*transpose_BLOCK_DIM + localIdX;
//      write the data to global memory
if ((i_index<box_nx)&&(j_index<box_ny)&&(k_index<box_nz))
{
t13_u_update1234[a3D_FinC((box_ny),(box_nx),(box_nz),k_index,j_index,i_index)]=block_u1234[localIdX][localIdY][localIdZ];
}
}
