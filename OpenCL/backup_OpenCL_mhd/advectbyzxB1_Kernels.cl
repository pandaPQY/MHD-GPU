__kernel 
void 
advectbyzxB1( 
	__global data_type_t* advB1_u,
	__global data_type_t* advB1_b,
	int advB1_nx,
	int advB1_ny,
	int advB1_nz,
	data_type_t advB1_dt,
	__global data_type_t* advB1_tmpB,
	__global data_type_t* advB1_u_update,
	__global data_type_t* advB1_b_update)
{     

__local data_type_t advB1_s_u[5*SIZEofWORKGROUP];
__local data_type_t advB1_s_u_km[5*SIZEofWORKGROUP];
__local data_type_t advB1_s_b[3*SIZEofWORKGROUP];

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
int advB1_km;
advB1_km=(groupIdY+(advB1_nz)-1)%(advB1_nz);
//
for (int ii=0; ii<5; ii++)
{
        advB1_s_u[a2D_FinC(5,localSizeX,ii,localIdX)]=advB1_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)];
}
for (int ii=0; ii<5; ii++)
{
        advB1_s_u_km[a2D_FinC(5,localSizeX,ii,localIdX)]=advB1_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,advB1_km)];
}
for (int ii=0; ii<3; ii++)
{
        advB1_s_b[a2D_FinC(3,localSizeX,ii,localIdX)]=advB1_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)];
}
barrier(CLK_LOCAL_MEM_FENCE);
//
data_type_t vx;
vx=(advB1_s_u_km[a2D_FinC(5,localSizeX,(2-1),localIdX)]+advB1_s_u[a2D_FinC(5,localSizeX,(2-1),localIdX)])/(advB1_s_u_km[a2D_FinC(5,localSizeX,(1-1),localIdX)]+advB1_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]);
//
int advB1_imm,advB1_imp;
advB1_imm=(localIdX+(advB1_nx)-1)%(advB1_nx);
advB1_imp=(localIdX+1)%(advB1_nx);
//
__local data_type_t advB1_s_tmp1[SIZEofWORKGROUP];
advB1_s_tmp1[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
vx=(advB1_s_tmp1[advB1_imm]+advB1_s_tmp1[advB1_imp]+2.0*advB1_s_tmp1[localIdX])/4.0;
//
data_type_t b1x;
b1x=advB1_s_b[a2D_FinC(3,localSizeX,(3-1),localIdX)];
//
//      second tvdb
data_type_t vg;
vg=vx;
data_type_t b;
b=b1x;
__local data_type_t advB1_s_vg[SIZEofWORKGROUP];
advB1_s_vg[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
data_type_t vh;
vh=(advB1_s_vg[localIdX]+advB1_s_vg[advB1_imp])/2.0;
//
__local data_type_t advB1_s_tmp2[SIZEofWORKGROUP];
advB1_s_tmp2[localIdX]=b*vg;
barrier(CLK_LOCAL_MEM_FENCE);
data_type_t flux1;
if (vh>0) flux1=b*vg;
else flux1=advB1_s_tmp2[advB1_imp];
advB1_s_tmp1[localIdX]=flux1;
barrier(CLK_LOCAL_MEM_FENCE);
data_type_t b1;
b1=b-(flux1-advB1_s_tmp1[advB1_imm])*(advB1_dt)/2.0;
//
int ip;
int ipp;
int im;
ip=(localIdX+1)%(advB1_nx);
ipp=(ip+1)%(advB1_nx);
im=(localIdX+(advB1_nx)-1)%(advB1_nx);
//
data_type_t v;
v=vh;
data_type_t w;
data_type_t wp;
data_type_t wm;
__local data_type_t advB1_s_b1_tvdb[SIZEofWORKGROUP];
advB1_s_b1_tvdb[localIdX]=b1;
barrier(CLK_LOCAL_MEM_FENCE);
if (v>0)
{
        w=advB1_s_vg[localIdX]*advB1_s_b1_tvdb[localIdX];
        wp=(advB1_s_vg[ip]*advB1_s_b1_tvdb[ip]-w)/2.0;
        wm=(w-advB1_s_vg[im]*advB1_s_b1_tvdb[im])/2.0;
}
else
{
        w=advB1_s_vg[ip]*advB1_s_b1_tvdb[ip];
        wp=(w-advB1_s_vg[ipp]*advB1_s_b1_tvdb[ipp])/2.0;
        wm=(advB1_s_vg[localIdX]*advB1_s_b1_tvdb[localIdX]-w)/2.0;
}
data_type_t dw;
dw=0.0;
//
if (wm*wp>0) dw=2.0*wm*wp/(wm+wp);
data_type_t flux;
flux=(w+dw)*(advB1_dt);
//
advB1_s_tmp2[localIdX]=flux;
barrier(CLK_LOCAL_MEM_FENCE);
b=b-(flux-advB1_s_tmp2[advB1_imm]);
//      finished tvdb
advB1_s_b[a2D_FinC(3,localSizeX,(3-1),localIdX)]=b;
advB1_s_b[a2D_FinC(3,localSizeX,(1-1),localIdX)]=advB1_s_b[a2D_FinC(3,localSizeX,(1-1),localIdX)]-advB1_s_tmp2[advB1_imm];
for (int ii=0; ii<3; ii++)
{
        advB1_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)]=advB1_s_b[a2D_FinC(3,localSizeX,ii,localIdX)];
}
	advB1_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=advB1_s_tmp2[advB1_imm];
//
}

