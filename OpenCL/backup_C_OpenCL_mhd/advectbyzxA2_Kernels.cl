__kernel 
void 
advectbyzxA2( 
	__global float4* advA2_u1234,
	__global float* advA2_u5,
	__global float4* advA2_b,
	int advA2_nx,
	int advA2_ny,
	int advA2_nz,
	float advA2_dt,
	__global float* advA2_tmpB)
{     

        uint globalIdX = get_global_id(0);
        uint globalIdY = get_global_id(1);

        uint groupIdX = get_group_id(0);
        uint groupIdY = get_group_id(1);

        uint localIdX = get_local_id(0);
        uint localIdY = get_local_id(1);

	uint localSizeX = get_local_size(0);
	uint localSizeY = get_local_size(1);

	uint numGroupsX = get_num_groups(0);
	uint numGroupsY = get_num_groups(1);

/*
	localIdX is i
	groupIdX is j
	groupIdY is k
	localSizeX is nx
	numGroupsX is ny
	numGroupsY is nz
*/
//	start
int advA2_jm;
advA2_jm=(groupIdX+(advA2_ny)-1)%(advA2_ny);
//
float4 tmpB_float4;
tmpB_float4.x=advA2_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
tmpB_float4.y=0.0;
tmpB_float4.z=0.0;
tmpB_float4.w=0.0;
//advA2_b.x[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,advA2_jm,groupIdY)]=advA2_b.x[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,advA2_jm,groupIdY)]+advA2_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
advA2_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,advA2_jm,groupIdY)]=advA2_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,advA2_jm,groupIdY)]+tmpB_float4;

}

