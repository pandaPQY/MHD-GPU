data_type_t max_3num(data_type_t *m3_1, data_type_t *m3_2, data_type_t *m3_3);

data_type_t max_2num(data_type_t *m2_1, data_type_t *m2_2);

__kernel 
void 
calcfl( 
	__global data_type_t* cfl_u,
	__global data_type_t* cfl_b,
	int cfl_nx,
	int cfl_ny,
	int cfl_nz,
	data_type_t cfl_dt,
	__global data_type_t* cfl_adv_tmpB,
	__global data_type_t* cfl_u_update,
	__global data_type_t* cfl_b_update,
	__global data_type_t* cfl_tmpC)
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
__local data_type_t cfl_s_b1_ip[SIZEofWORKGROUP];
__local data_type_t cfl_s_b2_jp[SIZEofWORKGROUP];
__local data_type_t cfl_s_b3_kp[SIZEofWORKGROUP];
__local data_type_t cfl_s_b[3*SIZEofWORKGROUP];
__local data_type_t cfl_s_u[5*SIZEofWORKGROUP];
data_type_t gamma;
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
cfl_s_b1_ip[localIdX]=cfl_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(1-1),ip,groupIdX,groupIdY)];
//      get cfl_s_b2_jp
cfl_s_b2_jp[localIdX]=cfl_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(2-1),localIdX,jp,groupIdY)];
//      get cfl_s_b3_kp
cfl_s_b3_kp[localIdX]=cfl_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,(3-1),localIdX,groupIdX,kp)];
//      get cfl_s_u
for (ii=0;ii<5;ii++)
{
        cfl_s_u[a2D_FinC(5,localSizeX,ii,localIdX)]=cfl_u[a4D_FinC(5,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)];
}
//      get cfl_s_b
for (ii=0;ii<3;ii++)
{
        cfl_s_b[a2D_FinC(3,localSizeX,ii,localIdX)]=cfl_b[a4D_FinC(3,localSizeX,numGroupsX,numGroupsY,ii,localIdX,groupIdX,groupIdY)];
}
//
barrier(CLK_LOCAL_MEM_FENCE);
//
data_type_t bx,by,bz;
//      bx=(b(1,i,j,k)+b(1,ip,j,k))/2
bx=(cfl_s_b[a2D_FinC(3,localSizeX,(1-1),localIdX)]+cfl_s_b1_ip[localIdX])/2.0;
//      by=(b(2,i,j,k)+b(2,i,jp,k))/2
by=(cfl_s_b[a2D_FinC(3,localSizeX,(2-1),localIdX)]+cfl_s_b2_jp[localIdX])/2.0;
//      bz=(b(3,i,j,k)+b(3,i,j,kp))/2
bz=(cfl_s_b[a2D_FinC(3,localSizeX,(3-1),localIdX)]+cfl_s_b3_kp[localIdX])/2.0;
data_type_t v;
//      v=maxval(abs(u(2:4,i,j,k)/u(1,i,j,k)))
data_type_t temp1,temp2,temp3;
temp1=fabs(cfl_s_u[a2D_FinC(5,localSizeX,(2-1),localIdX)]/cfl_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]);
temp2=fabs(cfl_s_u[a2D_FinC(5,localSizeX,(3-1),localIdX)]/cfl_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]);
temp3=fabs(cfl_s_u[a2D_FinC(5,localSizeX,(4-1),localIdX)]/cfl_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]);
v=max_3num(&temp1,&temp2,&temp3);
data_type_t b2;
b2=bx*bx+by*by+bz*bz;
//      ps=(u(5,i,j,k)-sum(u(2:4,i,j,k)**2,1)/u(1,i,j,k)/2)*(gamma-1)+(2-gamma)*b2/2
data_type_t ps;
ps=(cfl_s_u[a2D_FinC(5,localSizeX,(5-1),localIdX)]-(cfl_s_u[a2D_FinC(5,localSizeX,(2-1),localIdX)]*cfl_s_u[a2D_FinC(5,localSizeX,(2-1),localIdX)]+cfl_s_u[a2D_FinC(5,localSizeX,(3-1),localIdX)]*cfl_s_u[a2D_FinC(5,localSizeX,(3-1),localIdX)]+cfl_s_u[a2D_FinC(5,localSizeX,(4-1),localIdX)]*cfl_s_u[a2D_FinC(5,localSizeX,(4-1),localIdX)])/cfl_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]/2.0)*(gamma-1.0)+(2.0-gamma)*b2/2.0;
//      p=ps-b2/2
data_type_t p;
p=ps-b2/2.0;
//      c=max(c,v+sqrt(abs(  (b2*2+gamma*p)/u(1,i,j,k))))
temp2=cfl_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)];
temp3=(b2*2.0+gamma*p)/temp2;
temp3=fabs(temp3);
temp1=v+sqrt(temp3);
//temp1=v+sqrt(fabs((b2*2.0+gamma*p)/cfl_s_u[a2D_FinC(5,localSizeX,(1-1),localIdX)]));
//      find max
__local data_type_t cfl_s_c[SIZEofWORKGROUP];
data_type_t temp_c_max;
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

data_type_t max_3num(data_type_t *m3_1, data_type_t *m3_2, data_type_t *m3_3)
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

data_type_t max_2num(data_type_t *m2_1, data_type_t *m2_2)
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

