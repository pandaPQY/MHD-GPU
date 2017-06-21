__kernel 
void 
advectbyzxB2( 
	__global float4* advB2_u1234,
	__global float* advB2_u5,
	__global float4* advB2_b,
	int advB2_nx,
	int advB2_ny,
	int advB2_nz,
	float advB2_dt,
	__global float* advB2_tmpB)
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
int advB2_km;
advB2_km=(groupIdY+(advB2_nz)-1)%(advB2_nz);
float4 tmpB_float4;
tmpB_float4.x=advB2_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
tmpB_float4.y=0.0;
tmpB_float4.z=0.0;
tmpB_float4.w=0.0;

//
//advB2_b.x[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,advB2_km)]=advB2_b.x[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,advB2_km)]+advB2_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
advB2_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,advB2_km)]=advB2_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,advB2_km)]+tmpB_float4;
//
}

