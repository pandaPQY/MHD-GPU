! TVD split MHD code
! copyright (C) 2001,2003, Ue-Li Pen
! written November 2001 by Ue-Li Pen, pen@cita.utoronto.ca
!This program is free software; you can redistribute it and/or
!modify it under the terms of the GNU General Public License
!as published by the Free Software Foundation; either version 2
!of the License, or (at your option) any later version.
!
!This program is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!GNU General Public License for more details.
!
!You should have received a copy of the GNU General Public License
!along with this program; if not, write to the Free Software
!Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
!
! debugged to pass alven test March 29, 2002
!
! release 0.1 May 2003
! release 0.2 Sept 2009 : fixed b2 interpolation in cfl
!
! General Notes:
! restrictions: requires nx=nz or nz=1
! see http://arxiv.org/astroph/abs/astro-ph/0305088
! or http://www.cita.utoronto.ca/~pen/MHD
! for questions contact pen@cita.utoronto.ca
!
program main
use omp_lib
use works
!call omp_set_num_threads(16)
!call omp_set_nested(.true.)
!print*,'omp_is_initial_device()=',omp_is_initial_device()
!print*,'omp_get_num_devices()=',omp_get_num_devices()
!print*,'omp_get_initial_device()=',omp_get_initial_device()
!print*,'omp_get_default_device()=',omp_get_default_device()
!print*,'omp_get_num_threads()',omp_get_num_threads()
call init(u,b,nx,ny,nz)
print*,maxval(u),minval(u),maxval(b),minval(b)
tf=nx*4
t=0
iter=0
time=0
time2=0
print*,n
!$omp target data map(tofrom:u,b,t,dt,time,iter) map(to:tf,ostart,oend) 
!if( omp_is_initial_device() ) &
!  print*,'on the initial device'!   stop "not executing on target device"
do! iter=1,44
!call cpu_time(fstart)
if (t>=tf ) exit
!if (iter==3) exit
!print*,'t=',t
!$omp target! map(tofrom:u,b,t,dt,time,iter) map(to:tf,ostart,oend) 
!!$omp end target
!!$omp target
ostart=omp_get_wtime()
  iter=iter+1
!   if (t>=tf ) exit
    dt=0.9* cfl(u,b)!,nx,ny,nz)
   dt=min(dt,(tf-t)/2)
!   dt=0.5
   t=t+2*dt
   call fluidx(u,b,nx,ny,nz,dt)
   call advectbyzx(u,b,nx,ny,nz,dt)
!! the y sweep
   call transpose12(u,b,u,b,nx,ny,nz)
   call fluidx(u,b,ny,nx,nz,dt)
   call advectbyzx(u,b,ny,nx,nz,dt)
!! z sweep
   call transpose13(u,b,u,b,ny,nx,nz)
   call fluidx(u,b,nz,nx,ny,dt)
   call advectbyzx(u,b,nz,nx,ny,dt)
   call advectbyzx(u,b,nz,nx,ny,dt)
   call fluidx(u,b,nz,nx,ny,dt)
   
! back
   call transpose13(u,b,u,b,nz,nx,ny)
   call advectbyzx(u,b,ny,nx,nz,dt)
   call fluidx(u,b,ny,nx,nz,dt)
! x again
   call transpose12(u,b,u,b,ny,nx,nz)
   call advectbyzx(u,b,nx,ny,nz,dt)
   call fluidx(u,b,nx,ny,nz,dt)
!call cpu_time(fend)
oend=omp_get_wtime()
time=time+oend-ostart
!time2=time2+fend-fstart
!$omp end target
!$omp target update from(t)
!print*,'t=',t
end do
!!$omp end target
!!$omp end target 
!$omp end target data
print*,'n=',n,' iter=',iter,' t=',t,' dt=',dt
print*,maxval(u),minval(u),maxval(b),minval(b)
time=time/(iter-1)
time2=time2/iter

write(*,*) 'OpenMP Walltime elapsed', time*1000,' ms'
!write(*,*) 'Fortran CPU time elapsed', time2
!call output
end program main
