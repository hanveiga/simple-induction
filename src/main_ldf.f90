program main
  USE ISO_C_BINDING
  use parameters_dg_2d
  !real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,w,u_p
  real(kind=8),dimension(:,:,:,:,:),allocatable::u,w,u_p
  !device pointers
  type (c_ptr) :: u_d, du_d,dudt_d, w_d, u_eq_d, db_d
  type (c_ptr) :: x_d, y_d
  type (c_ptr) :: yc_d, xc_d
  !Initialization of fields
  call read_params()
  allocate(u(nx,ny,m,m,nvar),w(nx,ny,m,m,nvar),u_p(nx,ny,m,m,nvar))
  allocate(currentsol(nx,ny,m,m,nvar))
  allocate(x(nx,ny,m,m),y(nx,ny,m,m),xc(nx,ny),yc(nx,ny),w_eq(nx,ny,m,m,nvar),u_eq(nx,ny,m,m,nvar))
  allocate(x_quad(m),w_x_quad(m),y_quad(m),w_y_quad(m),x_gll(k),w_x_gll(k),y_gll(k),w_y_gll(k), sqrt_mod(m),sqrts_div(m))

  print*,'initialising u'
  call initialisation(u,w)
  !call get_equilibrium_solution()

  print*,'finish initialisign u'
  !device allocation and memory copies
  call devices()
  call setdevice(dev)
  call gpu_allocation(nvar,nx,ny,m,k,boxlen_x,boxlen_y,cfl,eta,bc,nequilibrium,gamma)
  call gpu_set_pointers(u_d,du_d,dudt_d,w_d,u_eq_d,x_d,y_d, xc_d, yc_d &
            & ,x_quad,y_quad,w_x_quad,w_y_quad,x_gll,y_gll,w_x_gll,w_y_gll,sqrt_mod)

  call gpu_set_more_pointers(sqrts_div, db_d)

  call h2d(u,u_d,nvar*nx*ny*m*m)
  call h2d(w,w_d,nvar*nx*ny*m*m)
  !call h2d(u_eq,u_eq_d,nvar*nx*ny*m*m)
  call h2d(x,x_d,nx*ny*m*m)
  call h2d(y,y_d,nx*ny*m*m)
  ! send cell centers
  call h2d(xc,xc_d,nx*ny)
  call h2d(yc,yc_d,nx*ny)

  call writedesc()

  if (ldf) then
    call evolve_ldf_test(u, u_d, du_d, db_d, dudt_d, u_eq_d, w_d)
  else
    call evolve(u, u_d, du_d, dudt_d, u_eq_d, w_d)
  end if

  !call device_get_modes_from_nodes_ldf(du_d,u_d)
  !call d2h(u_d,u,nvar*nx*ny*m*m) ! copy values back to host

  !call error_norm(u,tend,0)
  !call error_norm(u,tend,1)
  !call error_norm(u,tend,2)

end program main

subroutine evolve_ldf_test(u,u_d,du_d,db_d,dudt_d,u_eq_d,w_d)
  ! eat real nodal values, spit out nodal values
  USE ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: u_d, du_d, dudt_d, u_eq_d, w_d, db_d
  ! internal variables
  real(kind=8)::t,dt,top,start,finish,time
  real(kind=8)::cmax, dx, dy, cs_max,v_xmax,v_ymax
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,dudt,w1,w2,w3,w4,delta_u,nodes
  real(kind=8)::total_divB, surf_divB, vol_divB, vm
  integer::iter, n, i, j, irk, rkt, ssp
  integer::snap_counter
  integer::var


  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::nodes_cpu
  real(kind=8),dimension(1:nx,1:ny,1:(m)*(m+3)/2)::bmodes_cpu

  snap_counter = 0
  call writefields_new(snap_counter,u_d)

  snap_counter = 1
  call get_modes_from_nodes_cpu(u,bmodes_cpu)
  call get_nodes_from_modes_cpu(bmodes_cpu,nodes_cpu)
  call writefields_cpu(snap_counter,nodes_cpu)


  !do i = 1,50
  !  call get_modes_from_nodes_cpu(nodes_cpu,bmodes_cpu)
  !  call get_nodes_from_modes_cpu(bmodes_cpu,nodes_cpu)
  !  print*, 'looping'
  !end do
  !snap_counter = 2
  !call writefields_cpu(snap_counter,nodes_cpu)

  snap_counter = 1
  call device_get_modes_from_nodes_ldf_b_2(u_d,du_d, db_d)
  call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
  call writefields_new(snap_counter,u_d)

  !do i = 1,100
  !  call device_get_modes_from_nodes_ldf_b_2(u_d,du_d, db_d)
  !  call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
  !  print*, 'looping gpu'
  !end do
  !snap_counter = 2
  !call writefields_new(snap_counter,u_d)

  !open(12, file=trim(folder)//"/divB.txt", status="replace")
  !close(12)
  !call measure_divergence_ldf(total_divB, surf_divB, vol_divB,db_d)

  !call write_divergence(t,total_divB, surf_divB, vol_divB)
  !call write_divergence(t,total_divB, surf_divB, vol_divB)
  print*,'entered evolve'



  !if (wb) then
  t=0
  dtop=0.1
  top=dtop
  iter=0

  ssp=m

  if(ssp<4)then
     rkt= ssp-1
  else if(ssp>=4) then
     rkt = 4
     ssp = 4
  end if

  call cpu_time(start)
  do while(t <tend)
 !do while (iter<0)
      call device_compute_min_dt_t(dt)
      !call device_compute_max_v(vm)
      vm = 1.0
    
      dt = min(dt,tend-t)
      write(*,*)'time=',iter,t,dt
      do irk=0,rkt !RUNGE KUTTA SUBSTEPING
        call device_compute_update_lldf_test_new(irk,ssp,dt,t,vm)
      end do
      t=t+dt

      if ((make_movie).and.(t>=top))then
      !if (mod(iter,2)==0) then
         snap_counter = snap_counter + 1

         !call device_get_nodes_from_modes_ldf_b(du_d,u_d)
         call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
         call writefields_new(snap_counter,u_d)
         call measure_divergence_ldf(total_divB, surf_divB, vol_divB,db_d)
         call write_divergence(t,total_divB, surf_divB, vol_divB)

         top = top + dtop
      end if

      !if (t>=0.02) then
      !  dtop = 0.01
      !end if

      iter = iter + 1
   end do
   call cpu_time(finish)
   snap_counter = snap_counter + 1

   !call device_get_nodes_from_modes_ldf_b(du_d,u_d)
   call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
   call writefields_new(snap_counter,u_d)
   call measure_divergence_ldf(total_divB, surf_divB, vol_divB,db_d)
   call write_divergence(t,total_divB, surf_divB, vol_divB)

   print '("cpu average Time = ",e10.3," seconds.")',(finish-start)/dble(iter)
 !end if


  print*,'returned'
end subroutine evolve_ldf_test

subroutine evolve(u,u_d,du_d,dudt_d,u_eq_d,w_d)
  ! eat real nodal values, spit out nodal values
  USE ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: u_d, du_d, dudt_d, u_eq_d, w_d
  ! internal variables
  real(kind=8)::t,dt,top,start,finish,time
  real(kind=8)::cmax, dx, dy, cs_max,v_xmax,v_ymax
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,dudt,w1,w2,w3,w4,delta_u,nodes
  real(kind=8)::total_divB, surf_divB, vol_divB, vm
  integer::iter, n, i, j, irk, rkt, ssp
  integer::snap_counter
  integer::var

  !call device_compute_dt_grav();!It will only do something if the SRC is define over CUDA
  !call device_get_modes_from_nodes(u_eq_d,du_d)
  !call writefields(-1,du_d,delta_u)
  call device_get_modes_from_nodes(u_d,du_d)
  !call device_compute_limiter(du_d)
  snap_counter = 0
  t=0
  top=dtop
  iter=0
  ssp=m

  call writefields_new(snap_counter,du_d)

  open(12, file=trim(folder)//"/divB.txt", status="replace")
  close(12)
  call measure_divergence(total_divB, surf_divB, vol_divB)
  call write_divergence(t,total_divB, surf_divB, vol_divB)
  print*,'entered evolve'

  if(ssp<4)then
     rkt= 4
  else if(ssp>=4) then
     rkt = 4
     ssp = 4
  end if
  dtop = 0.0005
  top = dtop
  call cpu_time(start)
  do while(t < tend)
    ! Compute time step
    call device_compute_min_dt(dt)
    call device_compute_max_v(vm)
    dt = min(dt,tend-t)

    if (corr) then
      dt = min(dt,cfl/(2*(2*m-1))/(vm*vm*49.*nx))
      dt = dt/2.
    end if
    !write(*,*)'time=',vm,dt,cfl,m
    !dt = min(dt, (cfl/(2*(2*m-1))/(49.*(1.0+csx)/dx + 49.*(1.0+csy)/dy)))
    do irk=0,rkt !RUNGE KUTTA SUBSTEPING
      call device_compute_update(irk,ssp,dt,t,vm)
    end do

    if (corr) then
      do irk=0,rkt
         call post_process_b(irk,ssp,dt,t,vm)
      end do
      call parabolic_psi(dt,t,vm)
    end if

    if (corr) then
        t=t+2*dt
    else
        t = t + dt
    end if
    iter=iter+1
    write(*,*)'time=',iter,t,dt

    if (t>=0.02) then
      dtop = 0.01
    end if



    if ((make_movie).and.(t>=top))then
       snap_counter = snap_counter + 1
       call writefields_new(snap_counter,du_d)
       call measure_divergence(total_divB, surf_divB, vol_divB)
       call write_divergence(t,total_divB, surf_divB, vol_divB)
       top = top + dtop
    end if
    if (iter==1) then
      call writefields_new(-1,du_d)
    end if



 end do
 call cpu_time(finish)
 print '("cpu average Time = ",e10.3," seconds.")',(finish-start)/dble(iter)
end subroutine evolve


subroutine read_params
  USE ISO_C_BINDING
  use parameters_dg_2d
  !character(LEN=80)::infile, info_file
  character(LEN=32)::arg
  character(LEN=12)::cninit,cnx,cm
  character(LEN=20)::ceta

  !namelist/run_params/nx,ny,m,rk,boxlen_x,boxlen_y,gamma,cfl,tend
  !namelist/setup_params/bc,ninit
  !namelist/output_params/dtop,folder,dt_mon
  !namelist/source_params/xs,ys,rp,r_in,r_out,smp,Ms,Mp,grav
  ! Read namelist filename from command line argument
  !narg = iargc()
  !if(narg .lt. 1)then
  !   write(*,*)'File input.nml should contain a parameter namelist'
  !   stop
  !else
  !DO i = 1, iargc()
  !  CALL getarg(i, arg)
  !  WRITE (*,*) arg
  !END DO
  !n,nx,nquad,ninit,bc

  call getarg(1,arg)
  read(arg,*) nx

  call getarg(2,arg)
  read(arg,*) ny

  call getarg(3,arg)
  read(arg,*) m

  call getarg(4,arg)
  read(arg,*) ninit


  call getarg(5,arg)
  read(arg,*) folder


  k=(m+3)/2

  write (cnx,"(I2)") nx
  !folder = 'ic_'//trim(cninit)//'_'//trim(cnx)//'_'//trim(cm)//'_'//trim(ceta)
  !write (folder,"(A2,I0.2,A,I0.3,A,I0.1,A,I0.1,A4)") "ic_",ninit,'_',nx,'_',m,'_',bc,'_dcl'
  folder = "runs/"//trim(folder)
end subroutine read_params
