__kernel 
void 
transpose13( 
	__global data_type_t* t13_u,
	__global data_type_t* t13_b,
	int t13_nx,
	int t13_ny,
	int t13_nz,
	data_type_t t13_dt,
	__global data_type_t* t13_adv_tmpB,
	__global data_type_t* t13_u_update,
	__global data_type_t* t13_b_update)
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
	groupIdY is k
	groupIdX is j
	localIdX is i
	numGroupsY is nz
	numGroupsX is ny
	localSizeX is nx
*/
//	start
t13_u_update[a4D_FinC(5,numGroupsY,numGroupsX,localSizeX,(1-1),groupIdY,groupIdX,localIdX)]=t13_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(1-1),localIdX,groupIdX,groupIdY)];
t13_u_update[a4D_FinC(5,numGroupsY,numGroupsX,localSizeX,(2-1),groupIdY,groupIdX,localIdX)]=t13_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(4-1),localIdX,groupIdX,groupIdY)];
t13_u_update[a4D_FinC(5,numGroupsY,numGroupsX,localSizeX,(3-1),groupIdY,groupIdX,localIdX)]=t13_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(3-1),localIdX,groupIdX,groupIdY)];
t13_u_update[a4D_FinC(5,numGroupsY,numGroupsX,localSizeX,(4-1),groupIdY,groupIdX,localIdX)]=t13_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(2-1),localIdX,groupIdX,groupIdY)];
t13_u_update[a4D_FinC(5,numGroupsY,numGroupsX,localSizeX,(5-1),groupIdY,groupIdX,localIdX)]=t13_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(5-1),localIdX,groupIdX,groupIdY)];
//
t13_b_update[a4D_FinC(3,numGroupsY,numGroupsX,localSizeX,(1-1),groupIdY,groupIdX,localIdX)]=t13_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(3-1),localIdX,groupIdX,groupIdY)];
t13_b_update[a4D_FinC(3,numGroupsY,numGroupsX,localSizeX,(2-1),groupIdY,groupIdX,localIdX)]=t13_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(2-1),localIdX,groupIdX,groupIdY)];
t13_b_update[a4D_FinC(3,numGroupsY,numGroupsX,localSizeX,(3-1),groupIdY,groupIdX,localIdX)]=t13_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(1-1),localIdX,groupIdX,groupIdY)];

}

