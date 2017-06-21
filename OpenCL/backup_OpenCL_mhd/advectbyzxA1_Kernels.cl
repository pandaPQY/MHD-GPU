__kernel 
void 
advectbyzxA1( 
	__global data_type_t* advA1_u,
	__global data_type_t* advA1_b,
	int advA1_nx,
	int advA1_ny,
	int advA1_nz,
	data_type_t advA1_dt,
	__global data_type_t* advA1_tmpB,
	__global data_type_t* advA1_u_update,
	__global data_type_t* advA1_b_update,
	__global int* flu_fluidx_test)
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
__local data_type_t advA1_s_u[5*SIZEofWORKGROUP];
__local data_type_t advA1_s_u_jm[5*SIZEofWORKGROUP];
__local data_type_t advA1_s_b[3*SIZEofWORKGROUP];
//
int advA1_jm;
advA1_jm=(groupIdX+(advA1_ny)-1)%(advA1_ny);
//
for (int ii=0; ii<5; ii++)
{
        advA1_s_u[a2D_FinC(5,localSizeX,ii,localIdX)]=advA1_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)];
}
for (int ii=0; ii<5; ii++)
{
        advA1_s_u_jm[a2D_FinC(5,localSizeX,ii,localIdX)]=advA1_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,ii,localIdX,advA1_jm,groupIdY)];
}
for (int ii=0; ii<3; ii++)
{
        advA1_s_b[a2D_FinC(3,localSizeX,ii,localIdX)]=advA1_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)];
}
barrier(CLK_LOCAL_MEM_FENCE);
//
data_type_t vx;
vx=(advA1_s_u_jm[a2D_FinC(5,localSizeX,(2-1),localIdX)]+advA1_s_u[a2D_FinC(5,localSizeX,(2-1),localIdX)])/(advA1_s_u_jm[a2D_FinC(5,localSizeX,(1-1),localIdX)]+advA1_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]);
//
int advA1_imm,advA1_imp;
advA1_imm=(localIdX+(advA1_nx)-1)%(advA1_nx);
advA1_imp=(localIdX+1)%(advA1_nx);
//
__local data_type_t advA1_s_tmp1[SIZEofWORKGROUP];
advA1_s_tmp1[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
vx=(advA1_s_tmp1[advA1_imm]+advA1_s_tmp1[advA1_imp]+2.0*advA1_s_tmp1[localIdX])/4.0;
//
data_type_t b1x;
b1x=advA1_s_b[a2D_FinC(3,localSizeX,(2-1),localIdX)];
//
//      first tvdb
data_type_t vg;
vg=vx;
data_type_t b;
b=b1x;
__local data_type_t advA1_s_vg[SIZEofWORKGROUP];
advA1_s_vg[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
data_type_t vh;
vh=(advA1_s_vg[localIdX]+advA1_s_vg[advA1_imp])/2.0;
//
__local data_type_t advA1_s_tmp2[SIZEofWORKGROUP];
advA1_s_tmp2[localIdX]=b*vg;
barrier(CLK_LOCAL_MEM_FENCE);
data_type_t flux1;
if (vh>0) flux1=b*vg;
else flux1=advA1_s_tmp2[advA1_imp];
advA1_s_tmp1[localIdX]=flux1;
barrier(CLK_LOCAL_MEM_FENCE);
data_type_t b1;
b1=b-(flux1-advA1_s_tmp1[advA1_imm])*(advA1_dt)/2.0;
//
int ip;
int ipp;
int im;
ip=(localIdX+1)%(advA1_nx);
ipp=(ip+1)%(advA1_nx);
im=(localIdX+(advA1_nx)-1)%(advA1_nx);
//
data_type_t v;
v=vh;
data_type_t w;
data_type_t wp;
data_type_t wm;
__local data_type_t advA1_s_b1_tvdb[SIZEofWORKGROUP];
advA1_s_b1_tvdb[localIdX]=b1;
barrier(CLK_LOCAL_MEM_FENCE);
if (v>0)
{
        w=advA1_s_vg[localIdX]*advA1_s_b1_tvdb[localIdX];
        wp=(advA1_s_vg[ip]*advA1_s_b1_tvdb[ip]-w)/2.0;
        wm=(w-advA1_s_vg[im]*advA1_s_b1_tvdb[im])/2.0;
}
else
{
        w=advA1_s_vg[ip]*advA1_s_b1_tvdb[ip];
        wp=(w-advA1_s_vg[ipp]*advA1_s_b1_tvdb[ipp])/2.0;
        wm=(advA1_s_vg[localIdX]*advA1_s_b1_tvdb[localIdX]-w)/2.0;
}
data_type_t dw;
dw=0.0;
//
if (wm*wp>0) dw=2.0*wm*wp/(wm+wp);
data_type_t flux;
flux=(w+dw)*(advA1_dt);
//
advA1_s_tmp2[localIdX]=flux;
barrier(CLK_LOCAL_MEM_FENCE);
b=b-(flux-advA1_s_tmp2[advA1_imm]);
//      finished tvdb
//
advA1_s_b[a2D_FinC(3,localSizeX,(2-1),localIdX)]=b;
advA1_s_b[a2D_FinC(3,localSizeX,(1-1),localIdX)]=advA1_s_b[a2D_FinC(3,localSizeX,(1-1),localIdX)]-advA1_s_tmp2[advA1_imm];
//
//      send it back to global
for (int ii=0; ii<3; ii++)
{
        advA1_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)]=advA1_s_b[a2D_FinC(3,localSizeX,ii,localIdX)];
}
        advA1_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=advA1_s_tmp2[advA1_imm];
//
}

