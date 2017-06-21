void 
mhdflux(float *mf_v, float *mf_c, float *mf_u, float *mf_b, int *mf_n);

__kernel 
void 
fluidx( 
	__global float* flu_u,
	__global float* flu_b,
	int flu_nx,
	int flu_ny,
	int flu_nz,
	float flu_dt,
	__global float* flu_adv_tmpB,
	__global float* flu_u_update,
	__global float* flu_b_update,
	int WorkGroupSize,
	__local float* flu_s_u,
	__local float* flu_s_b,
	__local float* flu_s_jp_b2,
	__local float* flu_s_kp_b3,
	__local float* flu_s_b3x,
	__local float* mhdflux_max,
	__local float* s_c,
	__local float* tvd1_s_tmp1,
	__local float* tvd1_s_tmp2,
	__local float* tvd1_s_tmp3,
	__local float* tvd1_s_tmp4,
	__local float* tvd1_s_tmp5,
	__local float* tvd1_s_tmp6,
	__global int* flu_fluidx_test,
	__local float* mhdflux_max2,
	__local float* s_c2)
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

//      start
//
        flu_s_b[a2D_FinC(3,localSizeX,0,localIdX)]=flu_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,0,localIdX,groupIdX,groupIdY)];
        flu_s_b[a2D_FinC(3,localSizeX,1,localIdX)]=flu_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,1,localIdX,groupIdX,groupIdY)];
        flu_s_b[a2D_FinC(3,localSizeX,2,localIdX)]=flu_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,2,localIdX,groupIdX,groupIdY)];
//
        flu_s_u[a2D_FinC(5,localSizeX,0,localIdX)]=flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,0,localIdX,groupIdX,groupIdY)];
        flu_s_u[a2D_FinC(5,localSizeX,1,localIdX)]=flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,1,localIdX,groupIdX,groupIdY)];
        flu_s_u[a2D_FinC(5,localSizeX,2,localIdX)]=flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,2,localIdX,groupIdX,groupIdY)];
        flu_s_u[a2D_FinC(5,localSizeX,3,localIdX)]=flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,3,localIdX,groupIdX,groupIdY)];
        flu_s_u[a2D_FinC(5,localSizeX,4,localIdX)]=flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,4,localIdX,groupIdX,groupIdY)];
//
//
int flu_jp,flu_kp;
flu_jp=(groupIdX+1)%(flu_ny);
flu_kp=(groupIdY+1)%(flu_nz);
flu_s_jp_b2[localIdX]=flu_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(2-1),localIdX,flu_jp,groupIdY)];
flu_s_kp_b3[localIdX]=flu_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(3-1),localIdX,groupIdX,flu_kp)];
//


        flu_s_b3x[a2D_FinC(3,localSizeX,0,localIdX)]=flu_s_b[a2D_FinC(3,localSizeX,0,localIdX)]/2.0;
        flu_s_b3x[a2D_FinC(3,localSizeX,1,localIdX)]=flu_s_b[a2D_FinC(3,localSizeX,1,localIdX)]/2.0;
        flu_s_b3x[a2D_FinC(3,localSizeX,2,localIdX)]=flu_s_b[a2D_FinC(3,localSizeX,2,localIdX)]/2.0;
barrier(CLK_LOCAL_MEM_FENCE);
//
int flu_imp,flu_imm;
flu_imm=(localIdX+(flu_nx)-1)%(flu_nx);
flu_imp=(localIdX+1)%(flu_nx);
float flu_temp[3];
flu_temp[(1-1)]=flu_s_b3x[a2D_FinC(3,localSizeX,(1-1),flu_imp)];
flu_temp[(2-1)]=flu_s_jp_b2[localIdX]/2.0;
flu_temp[(3-1)]=flu_s_kp_b3[localIdX]/2.0;
//barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
barrier(CLK_LOCAL_MEM_FENCE);

        flu_s_b3x[a2D_FinC(3,localSizeX,0,localIdX)]=flu_s_b3x[a2D_FinC(3,localSizeX,0,localIdX)]+flu_temp[0];
        flu_s_b3x[a2D_FinC(3,localSizeX,1,localIdX)]=flu_s_b3x[a2D_FinC(3,localSizeX,1,localIdX)]+flu_temp[1];
        flu_s_b3x[a2D_FinC(3,localSizeX,2,localIdX)]=flu_s_b3x[a2D_FinC(3,localSizeX,2,localIdX)]+flu_temp[2];

barrier(CLK_LOCAL_MEM_FENCE);
//
// --- tvd1 part
//      first mhdflux
float tvd1_u[5], tvd1_b[3];
        tvd1_u[0]=flu_s_u[a2D_FinC(5,localSizeX,0,localIdX)];
        tvd1_u[1]=flu_s_u[a2D_FinC(5,localSizeX,1,localIdX)];
        tvd1_u[2]=flu_s_u[a2D_FinC(5,localSizeX,2,localIdX)];
        tvd1_u[3]=flu_s_u[a2D_FinC(5,localSizeX,3,localIdX)];
        tvd1_u[4]=flu_s_u[a2D_FinC(5,localSizeX,4,localIdX)];
        tvd1_b[0]=flu_s_b3x[a2D_FinC(3,localSizeX,0,localIdX)];
        tvd1_b[1]=flu_s_b3x[a2D_FinC(3,localSizeX,1,localIdX)];
        tvd1_b[2]=flu_s_b3x[a2D_FinC(3,localSizeX,2,localIdX)];
//
//
float v[5];
float thread_c;
int thread_n;
thread_n=localSizeX;
mhdflux(v,&thread_c,tvd1_u,tvd1_b,&thread_n);
//
mhdflux_max[localIdX]=thread_c;
barrier(CLK_LOCAL_MEM_FENCE);
//
int i_main;
if (localIdX==0)
{
float temp_c_max;
temp_c_max=0.0;
for (i_main=0; i_main<WorkGroupSize; i_main++)
{
        if (mhdflux_max[i_main]>temp_c_max) temp_c_max=mhdflux_max[i_main];
}
for (i_main=0; i_main<WorkGroupSize; i_main++)
{
        s_c[i_main]=temp_c_max;
}
}
barrier(CLK_LOCAL_MEM_FENCE);
//
float c;
c=s_c[localIdX];
//
if (c>0)
{
                v[0]=v[0]/c;
                v[1]=v[1]/c;
                v[2]=v[2]/c;
                v[3]=v[3]/c;
                v[4]=v[4]/c;
}
//
// --- tvd1 part 1
float wr[5];
        wr[0]=tvd1_u[0]+v[0];
        wr[1]=tvd1_u[1]+v[1];
        wr[2]=tvd1_u[2]+v[2];
        wr[3]=tvd1_u[3]+v[3];
        wr[4]=tvd1_u[4]+v[4];
float wl[5];
        wl[0]=tvd1_u[0]-v[0];
        wl[1]=tvd1_u[1]-v[1];
        wl[2]=tvd1_u[2]-v[2];
        wl[3]=tvd1_u[3]-v[3];
        wl[4]=tvd1_u[4]-v[4];
float fr[5];
        fr[0]=c*wr[0];
        fr[1]=c*wr[1];
        fr[2]=c*wr[2];
        fr[3]=c*wr[3];
        fr[4]=c*wr[4];
        tvd1_s_tmp1[a2D_FinC(5,localSizeX,0,localIdX)]=wl[0];
        tvd1_s_tmp1[a2D_FinC(5,localSizeX,1,localIdX)]=wl[1];
        tvd1_s_tmp1[a2D_FinC(5,localSizeX,2,localIdX)]=wl[2];
        tvd1_s_tmp1[a2D_FinC(5,localSizeX,3,localIdX)]=wl[3];
        tvd1_s_tmp1[a2D_FinC(5,localSizeX,4,localIdX)]=wl[4];
barrier(CLK_LOCAL_MEM_FENCE);
//
float fl[5];
        fl[0]=c*tvd1_s_tmp1[a2D_FinC(5,localSizeX,0,flu_imp)];
        fl[1]=c*tvd1_s_tmp1[a2D_FinC(5,localSizeX,1,flu_imp)];
        fl[2]=c*tvd1_s_tmp1[a2D_FinC(5,localSizeX,2,flu_imp)];
        fl[3]=c*tvd1_s_tmp1[a2D_FinC(5,localSizeX,3,flu_imp)];
        fl[4]=c*tvd1_s_tmp1[a2D_FinC(5,localSizeX,4,flu_imp)];
barrier(CLK_LOCAL_MEM_FENCE);
//
float flux[5];
        flux[0]=(fr[0]-fl[0])/2.0;
        flux[1]=(fr[1]-fl[1])/2.0;
        flux[2]=(fr[2]-fl[2])/2.0;
        flux[3]=(fr[3]-fl[3])/2.0;
        flux[4]=(fr[4]-fl[4])/2.0;
//
        tvd1_s_tmp2[a2D_FinC(5,localSizeX,0,localIdX)]=flux[0];
        tvd1_s_tmp2[a2D_FinC(5,localSizeX,1,localIdX)]=flux[1];
        tvd1_s_tmp2[a2D_FinC(5,localSizeX,2,localIdX)]=flux[2];
        tvd1_s_tmp2[a2D_FinC(5,localSizeX,3,localIdX)]=flux[3];
        tvd1_s_tmp2[a2D_FinC(5,localSizeX,4,localIdX)]=flux[4];
barrier(CLK_LOCAL_MEM_FENCE);
//
float tvd1_u1[5];
        tvd1_u1[0]=tvd1_u[0]-(flux[0]-tvd1_s_tmp2[a2D_FinC(5,localSizeX,0,flu_imm)])*(flu_dt)/2.0;
        tvd1_u1[1]=tvd1_u[1]-(flux[1]-tvd1_s_tmp2[a2D_FinC(5,localSizeX,1,flu_imm)])*(flu_dt)/2.0;
        tvd1_u1[2]=tvd1_u[2]-(flux[2]-tvd1_s_tmp2[a2D_FinC(5,localSizeX,2,flu_imm)])*(flu_dt)/2.0;
        tvd1_u1[3]=tvd1_u[3]-(flux[3]-tvd1_s_tmp2[a2D_FinC(5,localSizeX,3,flu_imm)])*(flu_dt)/2.0;
        tvd1_u1[4]=tvd1_u[4]-(flux[4]-tvd1_s_tmp2[a2D_FinC(5,localSizeX,4,flu_imm)])*(flu_dt)/2.0;
//
// --- mhdflux part 2
float v2[5];
float thread_c2;
float c2;
int i_main2;
mhdflux(v2,&thread_c2,tvd1_u1,tvd1_b,&thread_n);
//
mhdflux_max2[localIdX]=thread_c2;
barrier(CLK_LOCAL_MEM_FENCE);
if (localIdX==0)
{
float temp_c_max2;
temp_c_max2=0.0;
for (i_main2=0; i_main2<WorkGroupSize; i_main2++)
{
        if (mhdflux_max2[i_main2]>temp_c_max2) temp_c_max2=mhdflux_max2[i_main2];
}
for (i_main2=0; i_main2<WorkGroupSize; i_main2++)
{
        s_c2[i_main2]=temp_c_max2;
}
}
barrier(CLK_LOCAL_MEM_FENCE);
c2=s_c2[localIdX];
//
if (c2>0)
{
                v2[0]=v2[0]/c2;
                v2[1]=v2[1]/c2;
                v2[2]=v2[2]/c2;
                v2[3]=v2[3]/c2;
                v2[4]=v2[4]/c2;
}
//

// --- tvd1 part 2
        wr[0]=tvd1_u1[0]+v2[0];
        wr[1]=tvd1_u1[1]+v2[1];
        wr[2]=tvd1_u1[2]+v2[2];
        wr[3]=tvd1_u1[3]+v2[3];
        wr[4]=tvd1_u1[4]+v2[4];
        wl[0]=tvd1_u1[0]-v2[0];
        wl[1]=tvd1_u1[1]-v2[1];
        wl[2]=tvd1_u1[2]-v2[2];
        wl[3]=tvd1_u1[3]-v2[3];
        wl[4]=tvd1_u1[4]-v2[4];
        fr[0]=c2*wr[0];
        fr[1]=c2*wr[1];
        fr[2]=c2*wr[2];
        fr[3]=c2*wr[3];
        fr[4]=c2*wr[4];
        tvd1_s_tmp3[a2D_FinC(5,localSizeX,0,localIdX)]=fr[0];
        tvd1_s_tmp3[a2D_FinC(5,localSizeX,1,localIdX)]=fr[1];
        tvd1_s_tmp3[a2D_FinC(5,localSizeX,2,localIdX)]=fr[2];
        tvd1_s_tmp3[a2D_FinC(5,localSizeX,3,localIdX)]=fr[3];
        tvd1_s_tmp3[a2D_FinC(5,localSizeX,4,localIdX)]=fr[4];
barrier(CLK_LOCAL_MEM_FENCE);
//
float dfrp[5];
float dfrm[5];
float dfr[5];
        dfrp[0]=(tvd1_s_tmp3[a2D_FinC(5,localSizeX,0,flu_imp)]-fr[0])/2.0;
        dfrp[1]=(tvd1_s_tmp3[a2D_FinC(5,localSizeX,1,flu_imp)]-fr[1])/2.0;
        dfrp[2]=(tvd1_s_tmp3[a2D_FinC(5,localSizeX,2,flu_imp)]-fr[2])/2.0;
        dfrp[3]=(tvd1_s_tmp3[a2D_FinC(5,localSizeX,3,flu_imp)]-fr[3])/2.0;
        dfrp[4]=(tvd1_s_tmp3[a2D_FinC(5,localSizeX,4,flu_imp)]-fr[4])/2.0;
        dfrm[0]=(fr[0]-tvd1_s_tmp3[a2D_FinC(5,localSizeX,0,flu_imm)])/2.0;
        dfrm[1]=(fr[1]-tvd1_s_tmp3[a2D_FinC(5,localSizeX,1,flu_imm)])/2.0;
        dfrm[2]=(fr[2]-tvd1_s_tmp3[a2D_FinC(5,localSizeX,2,flu_imm)])/2.0;
        dfrm[3]=(fr[3]-tvd1_s_tmp3[a2D_FinC(5,localSizeX,3,flu_imm)])/2.0;
        dfrm[4]=(fr[4]-tvd1_s_tmp3[a2D_FinC(5,localSizeX,4,flu_imm)])/2.0;
        dfr[0]=0;
        dfr[1]=0;
        dfr[2]=0;
        dfr[3]=0;
        dfr[4]=0;
//
barrier(CLK_LOCAL_MEM_FENCE);
//
        if (dfrp[0]*dfrm[0]>0) dfr[0]=2.0*dfrp[0]*dfrm[0]/(dfrp[0]+dfrm[0]);
        if (dfrp[1]*dfrm[1]>0) dfr[1]=2.0*dfrp[1]*dfrm[1]/(dfrp[1]+dfrm[1]);
        if (dfrp[2]*dfrm[2]>0) dfr[2]=2.0*dfrp[2]*dfrm[2]/(dfrp[2]+dfrm[2]);
        if (dfrp[3]*dfrm[3]>0) dfr[3]=2.0*dfrp[3]*dfrm[3]/(dfrp[3]+dfrm[3]);
        if (dfrp[4]*dfrm[4]>0) dfr[4]=2.0*dfrp[4]*dfrm[4]/(dfrp[4]+dfrm[4]);
//
        tvd1_s_tmp4[a2D_FinC(5,localSizeX,0,localIdX)]=wl[0];
        tvd1_s_tmp4[a2D_FinC(5,localSizeX,1,localIdX)]=wl[1];
        tvd1_s_tmp4[a2D_FinC(5,localSizeX,2,localIdX)]=wl[2];
        tvd1_s_tmp4[a2D_FinC(5,localSizeX,3,localIdX)]=wl[3];
        tvd1_s_tmp4[a2D_FinC(5,localSizeX,4,localIdX)]=wl[4];
barrier(CLK_LOCAL_MEM_FENCE);
        fl[0]=c2*tvd1_s_tmp4[a2D_FinC(5,localSizeX,0,flu_imp)];
        fl[1]=c2*tvd1_s_tmp4[a2D_FinC(5,localSizeX,1,flu_imp)];
        fl[2]=c2*tvd1_s_tmp4[a2D_FinC(5,localSizeX,2,flu_imp)];
        fl[3]=c2*tvd1_s_tmp4[a2D_FinC(5,localSizeX,3,flu_imp)];
        fl[4]=c2*tvd1_s_tmp4[a2D_FinC(5,localSizeX,4,flu_imp)];
//
float dflp[5];
float dflm[5];
float dfl[5];
        tvd1_s_tmp5[a2D_FinC(5,localSizeX,0,localIdX)]=fl[0];
        tvd1_s_tmp5[a2D_FinC(5,localSizeX,1,localIdX)]=fl[1];
        tvd1_s_tmp5[a2D_FinC(5,localSizeX,2,localIdX)]=fl[2];
        tvd1_s_tmp5[a2D_FinC(5,localSizeX,3,localIdX)]=fl[3];
        tvd1_s_tmp5[a2D_FinC(5,localSizeX,4,localIdX)]=fl[4];
barrier(CLK_LOCAL_MEM_FENCE);
//
        dflp[0]=(fl[0]-tvd1_s_tmp5[a2D_FinC(5,localSizeX,0,flu_imp)])/2.0;
        dflp[1]=(fl[1]-tvd1_s_tmp5[a2D_FinC(5,localSizeX,1,flu_imp)])/2.0;
        dflp[2]=(fl[2]-tvd1_s_tmp5[a2D_FinC(5,localSizeX,2,flu_imp)])/2.0;
        dflp[3]=(fl[3]-tvd1_s_tmp5[a2D_FinC(5,localSizeX,3,flu_imp)])/2.0;
        dflp[4]=(fl[4]-tvd1_s_tmp5[a2D_FinC(5,localSizeX,4,flu_imp)])/2.0;
        dflm[0]=(tvd1_s_tmp5[a2D_FinC(5,localSizeX,0,flu_imm)]-fl[0])/2.0;
        dflm[1]=(tvd1_s_tmp5[a2D_FinC(5,localSizeX,1,flu_imm)]-fl[1])/2.0;
        dflm[2]=(tvd1_s_tmp5[a2D_FinC(5,localSizeX,2,flu_imm)]-fl[2])/2.0;
        dflm[3]=(tvd1_s_tmp5[a2D_FinC(5,localSizeX,3,flu_imm)]-fl[3])/2.0;
        dflm[4]=(tvd1_s_tmp5[a2D_FinC(5,localSizeX,4,flu_imm)]-fl[4])/2.0;
        dfl[0]=0;
        dfl[1]=0;
        dfl[2]=0;
        dfl[3]=0;
        dfl[4]=0;
//
barrier(CLK_LOCAL_MEM_FENCE);
//
        if (dflp[0]*dflm[0]>0) dfl[0]=2.0*dflp[0]*dflm[0]/(dflp[0]+dflm[0]);
        if (dflp[1]*dflm[1]>0) dfl[1]=2.0*dflp[1]*dflm[1]/(dflp[1]+dflm[1]);
        if (dflp[2]*dflm[2]>0) dfl[2]=2.0*dflp[2]*dflm[2]/(dflp[2]+dflm[2]);
        if (dflp[3]*dflm[3]>0) dfl[3]=2.0*dflp[3]*dflm[3]/(dflp[3]+dflm[3]);
        if (dflp[4]*dflm[4]>0) dfl[4]=2.0*dflp[4]*dflm[4]/(dflp[4]+dflm[4]);
//
        flux[0]=(fr[0]-fl[0]+(dfr[0]-dfl[0]))/2.0;
        flux[1]=(fr[1]-fl[1]+(dfr[1]-dfl[1]))/2.0;
        flux[2]=(fr[2]-fl[2]+(dfr[2]-dfl[2]))/2.0;
        flux[3]=(fr[3]-fl[3]+(dfr[3]-dfl[3]))/2.0;
        flux[4]=(fr[4]-fl[4]+(dfr[4]-dfl[4]))/2.0;
//
        tvd1_s_tmp6[a2D_FinC(5,localSizeX,0,localIdX)]=flux[0];
        tvd1_s_tmp6[a2D_FinC(5,localSizeX,1,localIdX)]=flux[1];
        tvd1_s_tmp6[a2D_FinC(5,localSizeX,2,localIdX)]=flux[2];
        tvd1_s_tmp6[a2D_FinC(5,localSizeX,3,localIdX)]=flux[3];
        tvd1_s_tmp6[a2D_FinC(5,localSizeX,4,localIdX)]=flux[4];
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
//

        flu_s_u[a2D_FinC(5,localSizeX,0,localIdX)]=flu_s_u[a2D_FinC(5,localSizeX,0,localIdX)]-(flux[0]-tvd1_s_tmp6[a2D_FinC(5,localSizeX,0,flu_imm)])*(flu_dt);
        flu_s_u[a2D_FinC(5,localSizeX,1,localIdX)]=flu_s_u[a2D_FinC(5,localSizeX,1,localIdX)]-(flux[1]-tvd1_s_tmp6[a2D_FinC(5,localSizeX,1,flu_imm)])*(flu_dt);
        flu_s_u[a2D_FinC(5,localSizeX,2,localIdX)]=flu_s_u[a2D_FinC(5,localSizeX,2,localIdX)]-(flux[2]-tvd1_s_tmp6[a2D_FinC(5,localSizeX,2,flu_imm)])*(flu_dt);
        flu_s_u[a2D_FinC(5,localSizeX,3,localIdX)]=flu_s_u[a2D_FinC(5,localSizeX,3,localIdX)]-(flux[3]-tvd1_s_tmp6[a2D_FinC(5,localSizeX,3,flu_imm)])*(flu_dt);
        flu_s_u[a2D_FinC(5,localSizeX,4,localIdX)]=flu_s_u[a2D_FinC(5,localSizeX,4,localIdX)]-(flux[4]-tvd1_s_tmp6[a2D_FinC(5,localSizeX,4,flu_imm)])*(flu_dt);
//
        flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,0,localIdX,groupIdX,groupIdY)]=flu_s_u[a2D_FinC(5,localSizeX,0,localIdX)];
        flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,1,localIdX,groupIdX,groupIdY)]=flu_s_u[a2D_FinC(5,localSizeX,1,localIdX)];
        flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,2,localIdX,groupIdX,groupIdY)]=flu_s_u[a2D_FinC(5,localSizeX,2,localIdX)];
        flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,3,localIdX,groupIdX,groupIdY)]=flu_s_u[a2D_FinC(5,localSizeX,3,localIdX)];
        flu_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,4,localIdX,groupIdX,groupIdY)]=flu_s_u[a2D_FinC(5,localSizeX,4,localIdX)];
// --- end tvd1
};

void mhdflux(float *mf_v, float *mf_c, float *mf_u, float *mf_b, int *mf_n)
{
float gamma;
gamma=5.0/3.0;
//
float vx;
vx=mf_u[(2-1)]/mf_u[(1-1)];
//
float ps;
ps=(mf_u[(5-1)]-(mf_u[(2-1)]*mf_u[(2-1)]+mf_u[(3-1)]*mf_u[(3-1)]+mf_u[(4-1)]*mf_u[(4-1)])/mf_u[(1-1)]/2.0)*(gamma-1.0)+(2.0-gamma)*(mf_b[(1-1)]*mf_b[(1-1)]+mf_b[(2-1)]*mf_b[(2-1)]+mf_b[(3-1)]*mf_b[(3-1)])/2.0;
//
mf_v[(1-1)]=mf_u[(2-1)];
mf_v[(2-1)]=mf_u[(2-1)]*vx+ps-mf_b[(1-1)]*mf_b[(1-1)];
mf_v[(3-1)]=mf_u[(3-1)]*vx-mf_b[(2-1)]*mf_b[(1-1)];
mf_v[(4-1)]=mf_u[(4-1)]*vx-mf_b[(3-1)]*mf_b[(1-1)];
mf_v[(5-1)]=(mf_u[(5-1)]+ps)*vx-mf_b[(1-1)]*(mf_b[(1-1)]*mf_u[(2-1)]+mf_b[(2-1)]*mf_u[(3-1)]+mf_b[(3-1)]*mf_u[(4-1)])/mf_u[(1-1)];
//
float p;
p=ps-(mf_b[(1-1)]*mf_b[(1-1)]+mf_b[(2-1)]*mf_b[(2-1)]+mf_b[(3-1)]*mf_b[(3-1)])/2.0;
//
(*mf_c)=fabs(vx)+sqrt(fabs((mf_b[(1-1)]*mf_b[(1-1)]+mf_b[(2-1)]*mf_b[(2-1)]+mf_b[(3-1)]*mf_b[(3-1)]+gamma*p)/mf_u[(1-1)]));
//
}

