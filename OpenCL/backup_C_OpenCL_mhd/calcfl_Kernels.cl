float max_3num(float *m3_1, float *m3_2, float *m3_3);

float max_2num(float *m2_1, float *m2_2);

__kernel 
void 
calcfl( 
	__global float4* cfl_u1234,
	__global float* cfl_u5,
	__global float4* cfl_b,
	int cfl_nx,
	int cfl_ny,
	int cfl_nz,
	float cfl_dt,
	__global float* cfl_tmpC)
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
float4 cfl_s_b1_ip;
float4 cfl_s_b2_jp;
float4 cfl_s_b3_kp;
float4 cfl_s_b;
float4 cfl_s_u1234;
float cfl_s_u5;
float gamma;
gamma=5.0/3.0;
int ii;
int kp,jp,ip;
//      kp=mod(k,nz)+1
kp=(groupIdY+1)%(box_nz);
//      jp=mod(j,ny)+1
jp=(groupIdX+1)%(box_ny);
//      ip=mod(i,nx)+1
ip=(localIdX+1)%(box_nx);
//      get cfl_s_b1_ip
cfl_s_b1_ip=cfl_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,ip,groupIdX,groupIdY)];
//      get cfl_s_b2_jp
cfl_s_b2_jp=cfl_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,jp,groupIdY)];
//      get cfl_s_b3_kp
cfl_s_b3_kp=cfl_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,kp)];
//      get cfl_s_u
        cfl_s_u1234=cfl_u1234[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
        cfl_s_u5=cfl_u5[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
//      get cfl_s_b
        cfl_s_b=cfl_b[a3D_FinC(localSizeX,numGroupsX,numGroupsY,localIdX,groupIdX,groupIdY)];
//
barrier(CLK_LOCAL_MEM_FENCE);
//
float bx,by,bz;
//      bx=(b(1,i,j,k)+b(1,ip,j,k))/2
bx=(cfl_s_b.x+cfl_s_b1_ip.x)*0.5;
//      by=(b(2,i,j,k)+b(2,i,jp,k))/2
by=(cfl_s_b.y+cfl_s_b2_jp.y)*0.5;
//      bz=(b(3,i,j,k)+b(3,i,j,kp))/2
bz=(cfl_s_b.z+cfl_s_b3_kp.z)*0.5;
float v;
//      v=maxval(abs(u(2:4,i,j,k)/u(1,i,j,k)))
float temp1,temp2,temp3;
temp1=fabs(cfl_s_u1234.y/cfl_s_u1234.x);
temp2=fabs(cfl_s_u1234.z/cfl_s_u1234.x);
temp3=fabs(cfl_s_u1234.w/cfl_s_u1234.x);
v=max_3num(&temp1,&temp2,&temp3);
float b2;
b2=bx*bx+by*by+bz*bz;
//      ps=(u(5,i,j,k)-sum(u(2:4,i,j,k)**2,1)/u(1,i,j,k)/2)*(gamma-1)+(2-gamma)*b2/2
float ps;
ps=(cfl_s_u5-(cfl_s_u1234.y*cfl_s_u1234.y+cfl_s_u1234.z*cfl_s_u1234.z+cfl_s_u1234.w*cfl_s_u1234.w)/cfl_s_u1234.x*0.5)*(gamma-1.0)+(2.0-gamma)*b2*0.5;
//      p=ps-b2/2
float p;
p=ps-b2*0.5;
//      c=max(c,v+sqrt(abs(  (b2*2+gamma*p)/u(1,i,j,k))))
temp2=cfl_s_u1234.x;
temp3=(b2*2.0+gamma*p)/temp2;
temp3=fabs(temp3);
temp1=v+sqrt(temp3);
//temp1=v+sqrt(fabs((b2*2.0+gamma*p)/cfl_s_u1234.x));
//      find max
__local float cfl_s_c[SIZEofWORKGROUP];
float temp_c_max;
cfl_s_c[localIdX]=temp1;
barrier(CLK_LOCAL_MEM_FENCE);
if (localIdX==0)
{
temp_c_max=0.0;
for (int i=0; i<SIZEofWORKGROUP; i++)
{
        if (cfl_s_c[i]>temp_c_max) temp_c_max=cfl_s_c[i];
}
cfl_tmpC[a2D_FinC(numGroupsX,numGroupsY,groupIdX,groupIdY)]=temp_c_max;
}
//
}

float max_3num(float *m3_1, float *m3_2, float *m3_3)
{
if ((*m3_1)>(*m3_2))
{
        if ((*m3_1)>(*m3_3))
        {
                return (*m3_1);
        }
        else
        {
                return (*m3_3);
        }
}
else
{
        if ((*m3_2)>(*m3_3))
        {
                return (*m3_2);
        }
        else
        {
                return (*m3_3);
        }
}
}

float max_2num(float *m2_1, float *m2_2)
{
if ((*m2_1)>(*m2_2))
{
        return (*m2_1);
}
else
{
        return (*m2_2);
}
}

