__kernel 
void 
advectbyzxB1( 
	__global float4* advB1_u1234,
	__global float* advB1_u5,
	__global float4* advB1_b,
	int advB1_nx,
	int advB1_ny,
	int advB1_nz,
	float advB1_dt,
	__global float* advB1_tmpB)
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
int advB1_km;
advB1_km=(groupIdY+(advB1_nz)-1)%(advB1_nz);
//
float4 advB1_s_u1234;
float4 advB1_s_u_km1234;
float4 advB1_s_b;
        advB1_s_u1234=advB1_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
        advB1_s_u_km1234=advB1_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,advB1_km)];
        advB1_s_b=advB1_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
barrier(CLK_LOCAL_MEM_FENCE);
//
float vx;
vx=(advB1_s_u_km1234.y+advB1_s_u1234.y)/(advB1_s_u_km1234.x+advB1_s_u1234.x);
//
int advB1_imm,advB1_imp;
advB1_imm=(localIdX+(advB1_nx)-1)%(advB1_nx);
advB1_imp=(localIdX+1)%(advB1_nx);
//
__local float advB1_s_tmp1[SIZEofWORKGROUP];
advB1_s_tmp1[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
vx=(advB1_s_tmp1[advB1_imm]+advB1_s_tmp1[advB1_imp]+2.0*advB1_s_tmp1[localIdX])*0.25;
//
float b1x;
b1x=advB1_s_b.z;
//
//      second tvdb
float vg;
vg=vx;
float b;
b=b1x;
__local float advB1_s_vg[SIZEofWORKGROUP];
advB1_s_vg[localIdX]=vx;
barrier(CLK_LOCAL_MEM_FENCE);
//
float vh;
vh=(advB1_s_vg[localIdX]+advB1_s_vg[advB1_imp])*0.5;
//
__local float advB1_s_tmp2[SIZEofWORKGROUP];
advB1_s_tmp2[localIdX]=b*vg;
barrier(CLK_LOCAL_MEM_FENCE);
float flux1;
if (vh>0) flux1=b*vg;
else flux1=advB1_s_tmp2[advB1_imp];
advB1_s_tmp1[localIdX]=flux1;
barrier(CLK_LOCAL_MEM_FENCE);
float b1;
b1=b-(flux1-advB1_s_tmp1[advB1_imm])*(advB1_dt)*0.5;
//
int ip;
int ipp;
int im;
ip=(localIdX+1)%(advB1_nx);
ipp=(ip+1)%(advB1_nx);
im=(localIdX+(advB1_nx)-1)%(advB1_nx);
//
float v;
v=vh;
float w;
float wp;
float wm;
__local float advB1_s_b1_tvdb[SIZEofWORKGROUP];
advB1_s_b1_tvdb[localIdX]=b1;
barrier(CLK_LOCAL_MEM_FENCE);
if (v>0)
{
        w=advB1_s_vg[localIdX]*advB1_s_b1_tvdb[localIdX];
        wp=(advB1_s_vg[ip]*advB1_s_b1_tvdb[ip]-w)*0.5;
        wm=(w-advB1_s_vg[im]*advB1_s_b1_tvdb[im])*0.5;
}
else
{
        w=advB1_s_vg[ip]*advB1_s_b1_tvdb[ip];
        wp=(w-advB1_s_vg[ipp]*advB1_s_b1_tvdb[ipp])*0.5;
        wm=(advB1_s_vg[localIdX]*advB1_s_b1_tvdb[localIdX]-w)*0.5;
}
float dw;
dw=0.0;
//
if (wm*wp>0) dw=2.0*wm*wp/(wm+wp);
float flux;
flux=(w+dw)*(advB1_dt);
//
advB1_s_tmp2[localIdX]=flux;
barrier(CLK_LOCAL_MEM_FENCE);
b=b-(flux-advB1_s_tmp2[advB1_imm]);
//      finished tvdb
advB1_s_b.z=b;
advB1_s_b.x=advB1_s_b.x-advB1_s_tmp2[advB1_imm];
        advB1_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=advB1_s_b;
	advB1_tmpB[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=advB1_s_tmp2[advB1_imm];
//
}

