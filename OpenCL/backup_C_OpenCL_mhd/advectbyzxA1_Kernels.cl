__kernel 
void 
advectbyzxA1( 
	__global float4* advA1_u1234,
	__global float* advA1_u5,
	__global float4* advA1_b,
	int advA1_nx,
	int advA1_ny,
	int advA1_nz,
	float advA1_dt,
	__global float* advA1_tmpB,
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
float4 advA1_s_u1234;
float4 advA1_s_u_jm1234;
float4 advA1_s_b;
//
int advA1_jm;
advA1_jm=(groupIdX+(advA1_ny)-1)%(advA1_ny);
//
        advA1_s_u1234=advA1_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
        advA1_s_u_jm1234=advA1_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,advA1_jm,groupIdY)];
        advA1_s_b=advA1_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
barrier(CLK_LOCAL_MEM_FENCE);
//
float vx;
vx=(advA1_s_u_jm1234.y+advA1_s_u1234.y)/(advA1_s_u_jm1234.x+advA1_s_u1234.x);
//
int advA1_imm,advA1_imp;
advA1_imm=(localIdX+(advA1_nx)-1)%(advA1_nx);
advA1_imp=(localIdX+1)%(advA1_nx);
//
__local float advA1_s_tmp1[SIZEofWORKGROUP];
advA1_s_tmp1[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
vx=(advA1_s_tmp1[advA1_imm]+advA1_s_tmp1[advA1_imp]+2.0*advA1_s_tmp1[localIdX])*0.25;
//
float b1x;
b1x=advA1_s_b.y;
//
//      first tvdb
float vg;
vg=vx;
float b;
b=b1x;
__local float advA1_s_vg[SIZEofWORKGROUP];
advA1_s_vg[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
float vh;
vh=(advA1_s_vg[localIdX]+advA1_s_vg[advA1_imp])*0.5;
//
__local float advA1_s_tmp2[SIZEofWORKGROUP];
advA1_s_tmp2[localIdX]=b*vg;
barrier(CLK_LOCAL_MEM_FENCE);
float flux1;
if (vh>0) flux1=b*vg;
else flux1=advA1_s_tmp2[advA1_imp];
advA1_s_tmp1[localIdX]=flux1;
barrier(CLK_LOCAL_MEM_FENCE);
float b1;
b1=b-(flux1-advA1_s_tmp1[advA1_imm])*(advA1_dt)*0.5;
//
int ip;
int ipp;
int im;
ip=(localIdX+1)%(advA1_nx);
ipp=(ip+1)%(advA1_nx);
im=(localIdX+(advA1_nx)-1)%(advA1_nx);
//
float v;
v=vh;
float w;
float wp;
float wm;
__local float advA1_s_b1_tvdb[SIZEofWORKGROUP];
advA1_s_b1_tvdb[localIdX]=b1;
barrier(CLK_LOCAL_MEM_FENCE);
if (v>0)
{
        w=advA1_s_vg[localIdX]*advA1_s_b1_tvdb[localIdX];
        wp=(advA1_s_vg[ip]*advA1_s_b1_tvdb[ip]-w)*0.5;
        wm=(w-advA1_s_vg[im]*advA1_s_b1_tvdb[im])*0.5;
}
else
{
        w=advA1_s_vg[ip]*advA1_s_b1_tvdb[ip];
        wp=(w-advA1_s_vg[ipp]*advA1_s_b1_tvdb[ipp])*0.5;
        wm=(advA1_s_vg[localIdX]*advA1_s_b1_tvdb[localIdX]-w)*0.5;
}
float dw;
dw=0.0;
//
if (wm*wp>0) dw=2.0*wm*wp/(wm+wp);
float flux;
flux=(w+dw)*(advA1_dt);
//
advA1_s_tmp2[localIdX]=flux;
barrier(CLK_LOCAL_MEM_FENCE);
b=b-(flux-advA1_s_tmp2[advA1_imm]);
//      finished tvdb
//
advA1_s_b.y=b;
advA1_s_b.x=advA1_s_b.x-advA1_s_tmp2[advA1_imm];
//
//      send it back to global
        advA1_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=advA1_s_b;
        advA1_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=advA1_s_tmp2[advA1_imm];
//
}

