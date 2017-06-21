__kernel 
void 
transpose12( 
	__global data_type_t* t12_u,
	__global data_type_t* t12_b,
	int t12_nx,
	int t12_ny,
	int t12_nz,
	data_type_t t12_dt,
	__global data_type_t* t12_adv_tmpB,
	__global data_type_t* t12_u_update,
	__global data_type_t* t12_b_update)
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

t12_u_update[a4D_FinC(5,numGroupsX,localSizeX,numGroupsY,(1-1),groupIdX,localIdX,groupIdY)]=t12_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(1-1),localIdX,groupIdX,groupIdY)];
t12_u_update[a4D_FinC(5,numGroupsX,localSizeX,numGroupsY,(2-1),groupIdX,localIdX,groupIdY)]=t12_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(3-1),localIdX,groupIdX,groupIdY)];
t12_u_update[a4D_FinC(5,numGroupsX,localSizeX,numGroupsY,(3-1),groupIdX,localIdX,groupIdY)]=t12_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(2-1),localIdX,groupIdX,groupIdY)];
t12_u_update[a4D_FinC(5,numGroupsX,localSizeX,numGroupsY,(4-1),groupIdX,localIdX,groupIdY)]=t12_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(4-1),localIdX,groupIdX,groupIdY)];
t12_u_update[a4D_FinC(5,numGroupsX,localSizeX,numGroupsY,(5-1),groupIdX,localIdX,groupIdY)]=t12_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(5-1),localIdX,groupIdX,groupIdY)];
//
t12_b_update[a4D_FinC(3,numGroupsX,localSizeX,numGroupsY,(1-1),groupIdX,localIdX,groupIdY)]=t12_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(2-1),localIdX,groupIdX,groupIdY)];
t12_b_update[a4D_FinC(3,numGroupsX,localSizeX,numGroupsY,(2-1),groupIdX,localIdX,groupIdY)]=t12_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(1-1),localIdX,groupIdX,groupIdY)];
t12_b_update[a4D_FinC(3,numGroupsX,localSizeX,numGroupsY,(3-1),groupIdX,localIdX,groupIdY)]=t12_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(3-1),localIdX,groupIdX,groupIdY)];

//t12_u_update[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(5-1),localIdX,groupIdX,groupIdY)]=t12_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,(5-1),localIdX,groupIdX,groupIdY)]+0.1;
}

