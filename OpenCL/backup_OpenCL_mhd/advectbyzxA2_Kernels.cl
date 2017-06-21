__kernel 
void 
advectbyzxA2( 
	__global data_type_t* advA2_u,
	__global data_type_t* advA2_b,
	int advA2_nx,
	int advA2_ny,
	int advA2_nz,
	data_type_t advA2_dt,
	__global data_type_t* advA2_tmpB,
	__global data_type_t* advA2_u_update,
	__global data_type_t* advA2_b_update)
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
advA2_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(1-1),localIdX,advA2_jm,groupIdY)]=advA2_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(1-1),localIdX,advA2_jm,groupIdY)]+advA2_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];

}

