__kernel 
void 
fluidx( 
	__global float4* flu_u1234,
	__global float* flu_u5,
	__global float4* flu_b,
	int flu_nx,
	int flu_ny,
	int flu_nz,
	float flu_dt,
	__global float* flu_adv_tmpB,
	int WorkGroupSize,
	__global int* flu_fluidx_test,
	float flu_CFL_value)
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
float4 flu_s_u1234;
float flu_s_u5;
flu_s_u1234=flu_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
flu_s_u5=flu_u5[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
//
float4 flu_s_jp_b,flu_s_kp_b;
float flu_s_jp_b2,flu_s_kp_b3;
int flu_jp,flu_kp;
flu_jp=(groupIdX+1)%(flu_ny);
flu_kp=(groupIdY+1)%(flu_nz);
flu_s_jp_b=flu_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,flu_jp,groupIdY)];
flu_s_kp_b=flu_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,flu_kp)];
flu_s_jp_b2=flu_s_jp_b.y;
flu_s_kp_b3=flu_s_kp_b.z;
//
float4 flu_s_b;
flu_s_b=flu_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
//
int flu_imp,flu_imm;
flu_imm=(localIdX+(flu_nx)-1)%(flu_nx);
flu_imp=(localIdX+1)%(flu_nx);
float4 flu_s_b_b3x;
flu_s_b_b3x=flu_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,flu_imp,groupIdX,groupIdY)];
float4 tvd1_b;
tvd1_b.x=flu_s_b.x*0.5+flu_s_b_b3x.x*0.5;
tvd1_b.y=flu_s_b.y*0.5+flu_s_jp_b2*0.5;
tvd1_b.z=flu_s_b.z*0.5+flu_s_kp_b3*0.5;
//
// --- tvd1 part
//      first mhdflux
float4 tvd1_u1234;
float tvd1_u5;
tvd1_u1234=flu_s_u1234;
tvd1_u5=flu_s_u5;
//
float4 v1234;
float v5;
//	start mhdflux
float gamma;
gamma=5.0/3.0;
float vx;
vx=(tvd1_u1234.y)/(tvd1_u1234.x);
float ps;
ps=(tvd1_u5-((tvd1_u1234.y)*(tvd1_u1234.y)+(tvd1_u1234.z)*(tvd1_u1234.z)+(tvd1_u1234.w)*(tvd1_u1234.w))/(tvd1_u1234.x)*0.5)*(gamma-1.0)+(2.0-gamma)*((tvd1_b.x)*(tvd1_b.x)+(tvd1_b.y)*(tvd1_b.y)+(tvd1_b.z)*(tvd1_b.z))*0.5;
(v1234.x)=(tvd1_u1234.y);
(v1234.y)=(tvd1_u1234.y)*vx+ps-(tvd1_b.x)*(tvd1_b.x);
(v1234.z)=(tvd1_u1234.z)*vx-(tvd1_b.y)*(tvd1_b.x);
(v1234.w)=(tvd1_u1234.w)*vx-(tvd1_b.z)*(tvd1_b.x);
(v5)=((tvd1_u5)+ps)*vx-(tvd1_b.x)*((tvd1_b.x)*(tvd1_u1234.y)+(tvd1_b.y)*(tvd1_u1234.z)+(tvd1_b.z)*(tvd1_u1234.w))/(tvd1_u1234.x);
//      finish mhdflux
//
float c;
c=flu_CFL_value;
//
if (c>0)
{
                v1234=v1234/c;
                v5=v5/c;
}
//
// --- tvd1 part 1
float4 wr1234;
float wr5;
        wr1234=tvd1_u1234+v1234;
        wr5=tvd1_u5+v5;
float4 wl1234;
float wl5;
        wl1234=tvd1_u1234-v1234;
        wl5=tvd1_u5-v5;
float4 fr1234;
float fr5;
        fr1234=c*wr1234;
        fr5=c*wr5;
//
__local float4 tvd1_Vs_tmp1_1234[SIZEofWORKGROUP]; 
__local float tvd1_Vs_tmp1_5[SIZEofWORKGROUP]; 
tvd1_Vs_tmp1_1234[localIdX]=wl1234;
tvd1_Vs_tmp1_5[localIdX]=wl5;
barrier(CLK_LOCAL_MEM_FENCE);
//
float4 fl1234;
float fl5;
fl1234=c*tvd1_Vs_tmp1_1234[flu_imp];
fl5=c*tvd1_Vs_tmp1_5[flu_imp];
//
float4 flux1234;
float flux5;
        flux1234=(fr1234-fl1234)*0.5;
        flux5=(fr5-fl5)*0.5;
//
__local float4 tvd1_Vs_tmp2_1234[SIZEofWORKGROUP]; 
__local float tvd1_Vs_tmp2_5[SIZEofWORKGROUP]; 
//
tvd1_Vs_tmp2_1234[localIdX]=flux1234;
tvd1_Vs_tmp2_5[localIdX]=flux5;
barrier(CLK_LOCAL_MEM_FENCE);
//
float4 tvd1_u1_1234;
float tvd1_u1_5;
        tvd1_u1_1234=tvd1_u1234-(flux1234-tvd1_Vs_tmp2_1234[flu_imm])*(flu_dt)*0.5;
        tvd1_u1_5=tvd1_u5-(flux5-tvd1_Vs_tmp2_5[flu_imm])*(flu_dt)*0.5;
//
// --- mhdflux part 2
float4 v2_1234;
float v2_5;
float c2;
int i_main2;
//      start second mhdflux
float vx2;
vx2=(tvd1_u1_1234.y)/(tvd1_u1_1234.x);
float ps2;
ps2=(tvd1_u1_5-((tvd1_u1_1234.y)*(tvd1_u1_1234.y)+(tvd1_u1_1234.z)*(tvd1_u1_1234.z)+(tvd1_u1_1234.w)*(tvd1_u1_1234.w))/(tvd1_u1_1234.x)*0.5)*(gamma-1.0)+(2.0-gamma)*((tvd1_b.x)*(tvd1_b.x)+(tvd1_b.y)*(tvd1_b.y)+(tvd1_b.z)*(tvd1_b.z))*0.5;
(v2_1234.x)=(tvd1_u1_1234.y);
(v2_1234.y)=(tvd1_u1_1234.y)*vx2+ps2-(tvd1_b.x)*(tvd1_b.x);
(v2_1234.z)=(tvd1_u1_1234.z)*vx2-(tvd1_b.y)*(tvd1_b.x);
(v2_1234.w)=(tvd1_u1_1234.w)*vx2-(tvd1_b.z)*(tvd1_b.x);
(v2_5)=((tvd1_u1_5)+ps2)*vx2-(tvd1_b.x)*((tvd1_b.x)*(tvd1_u1_1234.y)+(tvd1_b.y)*(tvd1_u1_1234.z)+(tvd1_b.z)*(tvd1_u1_1234.w))/(tvd1_u1_1234.x);
//
//      finish mhdflux
c2=flu_CFL_value;
//
if (c2>0)
{
                v2_1234=v2_1234/c2;
                v2_5=v2_5/c2;
}
//
// --- tvd1 part 2
        wr1234=tvd1_u1_1234+v2_1234;
        wr5=tvd1_u1_5+v2_5;
        wl1234=tvd1_u1_1234-v2_1234;
        wl5=tvd1_u1_5-v2_5;
        fr1234=c2*wr1234;
        fr5=c2*wr5;
//
__local float4 tvd1_Vs_tmp3_1234[SIZEofWORKGROUP];
__local float tvd1_Vs_tmp3_5[SIZEofWORKGROUP];
tvd1_Vs_tmp3_1234[localIdX]=fr1234;
tvd1_Vs_tmp3_5[localIdX]=fr5;
barrier(CLK_LOCAL_MEM_FENCE);
//
float4 dfrp1234;
float dfrp5;
float4 dfrm1234;
float dfrm5;
float4 dfr1234;
float dfr5;
        dfrp1234=(tvd1_Vs_tmp3_1234[flu_imp]-fr1234)*0.5;
        dfrp5=(tvd1_Vs_tmp3_5[flu_imp]-fr5)*0.5;
        dfrm1234=(fr1234-tvd1_Vs_tmp3_1234[flu_imm])*0.5;
        dfrm5=(fr5-tvd1_Vs_tmp3_5[flu_imm])*0.5;
        dfr1234=(float4)(0.0f,0.0f,0.0f,0.0f);
        dfr5=0.0;
barrier(CLK_LOCAL_MEM_FENCE);
//
        if (dfrp1234.x*dfrm1234.x>0) dfr1234.x=2.0*dfrp1234.x*dfrm1234.x/(dfrp1234.x+dfrm1234.x);
        if (dfrp1234.y*dfrm1234.y>0) dfr1234.y=2.0*dfrp1234.y*dfrm1234.y/(dfrp1234.y+dfrm1234.y);
        if (dfrp1234.z*dfrm1234.z>0) dfr1234.z=2.0*dfrp1234.z*dfrm1234.z/(dfrp1234.z+dfrm1234.z);
        if (dfrp1234.w*dfrm1234.w>0) dfr1234.w=2.0*dfrp1234.w*dfrm1234.w/(dfrp1234.w+dfrm1234.w);
        if (dfrp5*dfrm5>0) dfr5=2.0*(dfrp5)*(dfrm5)/(dfrp5+dfrm5);
//
__local float4 tvd1_Vs_tmp4_1234[SIZEofWORKGROUP];
__local float tvd1_Vs_tmp4_5[SIZEofWORKGROUP];
tvd1_Vs_tmp4_1234[localIdX]=wl1234;
tvd1_Vs_tmp4_5[localIdX]=wl5;
barrier(CLK_LOCAL_MEM_FENCE);
//
        fl1234=c2*tvd1_Vs_tmp4_1234[flu_imp];
        fl5=c2*tvd1_Vs_tmp4_5[flu_imp];
//
float4 dflp1234;
float dflp5;
float4 dflm1234;
float dflm5;
float4 dfl1234;
float dfl5;
//
__local float4 tvd1_Vs_tmp5_1234[SIZEofWORKGROUP];
__local float tvd1_Vs_tmp5_5[SIZEofWORKGROUP];
tvd1_Vs_tmp5_1234[localIdX]=fl1234;
tvd1_Vs_tmp5_5[localIdX]=fl5;
barrier(CLK_LOCAL_MEM_FENCE);
//
        dflp1234=(fl1234-tvd1_Vs_tmp5_1234[flu_imp])*0.5;
        dflp5=(fl5-tvd1_Vs_tmp5_5[flu_imp])*0.5;
        dflm1234=(tvd1_Vs_tmp5_1234[flu_imm]-fl1234)*0.5;
        dflm5=(tvd1_Vs_tmp5_5[flu_imm]-fl5)*0.5;
        dfl1234=(float4)(0.0f,0.0f,0.0f,0.0f);
        dfl5=0.0;
barrier(CLK_LOCAL_MEM_FENCE);
//
        if (dflp1234.x*dflm1234.x>0) dfl1234.x=2.0*dflp1234.x*dflm1234.x/(dflp1234.x+dflm1234.x);
        if (dflp1234.y*dflm1234.y>0) dfl1234.y=2.0*dflp1234.y*dflm1234.y/(dflp1234.y+dflm1234.y);
        if (dflp1234.z*dflm1234.z>0) dfl1234.z=2.0*dflp1234.z*dflm1234.z/(dflp1234.z+dflm1234.z);
        if (dflp1234.w*dflm1234.w>0) dfl1234.w=2.0*dflp1234.w*dflm1234.w/(dflp1234.w+dflm1234.w);
        if (dflp5*dflm5>0) dfl5=2.0*dflp5*dflm5/(dflp5+dflm5);
//
	flux1234=(fr1234-fl1234+(dfr1234-dfl1234))*0.5;
        flux5=(fr5-fl5+(dfr5-dfl5))*0.5;
//
__local float4 tvd1_Vs_tmp6_1234[SIZEofWORKGROUP];
__local float tvd1_Vs_tmp6_5[SIZEofWORKGROUP];
tvd1_Vs_tmp6_1234[localIdX]=flux1234;
tvd1_Vs_tmp6_5[localIdX]=flux5;
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
//
        flu_s_u1234=flu_s_u1234-(flux1234-tvd1_Vs_tmp6_1234[flu_imm])*(flu_dt);
        flu_s_u5=flu_s_u5-(flux5-tvd1_Vs_tmp6_5[flu_imm])*(flu_dt);
//
flu_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=flu_s_u1234;
flu_u5[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)]=flu_s_u5;
// --- end tvd1
};

