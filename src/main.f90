program main
  USE ISO_C_BINDING
  use parameters_dg_2d
  real(kind=8),dimension(:,:,:,:,:),allocatable::u,w,u_p

  !device pointers
  type (c_ptr) :: u_d, du_d,dudt_d, w_d, db_d
  type (c_ptr) :: x_d, y_d
  type (c_ptr) :: yc_d, xc_d

  call read_params()

  allocate(u(nx,ny,m,m,nvar),w(nx,ny,m,m,nvar),u_p(nx,ny,m,m,nvar))
  allocate(currentsol(nx,ny,m,m,nvar))
  allocate(x(nx,ny,m,m),y(nx,ny,m,m),xc(nx,ny),yc(nx,ny))
  allocate(x_quad(m),w_x_quad(m),y_quad(m),w_y_quad(m),x_gll(k),w_x_gll(k),y_gll(k),w_y_gll(k), sqrt_mod(m),sqrts_div(m))

  ! Initialization of fields
  call initialisation(u,w)

  !device allocation and memory copies
  call devices()
  call setdevice(dev)
  call gpu_allocation(nvar,nx,ny,m,k,boxlen_x,boxlen_y,cfl,eta,bc,gamma)
  call gpu_set_pointers(u_d,du_d,dudt_d,w_d,x_d,y_d, xc_d, yc_d &
            & ,x_quad,y_quad,w_x_quad,w_y_quad,x_gll,y_gll,w_x_gll,w_y_gll,sqrt_mod)

  call gpu_set_more_pointers(sqrts_div, db_d)

  call h2d(u,u_d,nvar*nx*ny*m*m)
  call h2d(w,w_d,nvar*nx*ny*m*m)
  call h2d(x,x_d,nx*ny*m*m)
  call h2d(y,y_d,nx*ny*m*m)

  ! send cell centers
  call h2d(xc,xc_d,nx*ny)
  call h2d(yc,yc_d,nx*ny)

  call writedesc()

  if (ldf) then
    call evolve_ldf(u, u_d, du_d, db_d, dudt_d, w_d)
  else
    call evolve(u, u_d, du_d, dudt_d, w_d)
  end if

end program main

subroutine evolve_ldf(u,u_d,du_d,db_d,dudt_d,w_d)
  USE ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: u_d, du_d, dudt_d, w_d, db_d
  ! internal variables
  real(kind=8)::t,dt,top,start,finish,time
  real(kind=8)::cmax, dx, dy, cs_max,v_xmax,v_ymax
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,dudt,delta_u,nodes
  real(kind=8)::total_divB, surf_divB, vol_divB, vm
  integer::iter, n, i, j, irk, rkt, ssp
  integer::snap_counter
  integer::var

  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::nodes_cpu
  real(kind=8),dimension(1:nx,1:ny,1:(m)*(m+3)/2)::bmodes_cpu

  snap_counter = 0
  ! IC
  call writefields_new(snap_counter,u_d)

  ! IC after projected into modes
  snap_counter = 1
  call device_get_modes_from_nodes_ldf_b_2(u_d,du_d, db_d)
  call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
  call writefields_new(snap_counter,u_d)

  t=0
  dtop=0.1
  top=dtop
  iter=0

  ssp=m

  if(ssp<4)then
     rkt= 4
     ssp = 4
  else if(ssp>=4) then
     rkt = 4
     ssp = 4
  end if

  call cpu_time(start)
  do while(t <tend)
      call device_compute_min_dt_t(dt)
      vm = 1.0

      dt = min(dt,tend-t)
      write(*,*)'time=',iter,t,dt
      do irk=0,rkt !RUNGE KUTTA SUBSTEPING
        call device_compute_update_lldf_test_new(irk,ssp,dt,t,vm)
      end do
      t=t+dt

      if ((make_movie).and.(t>=top))then
         snap_counter = snap_counter + 1

         call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
         call writefields_new(snap_counter,u_d)
         call measure_divergence_ldf(total_divB, surf_divB, vol_divB,db_d)
         call write_divergence(t,total_divB, surf_divB, vol_divB)

         top = top + dtop
      end if

      iter = iter + 1
   end do
   call cpu_time(finish)
   snap_counter = snap_counter + 1

   call device_get_nodes_from_modes_ldf_b_2(du_d,db_d,u_d)
   call writefields_new(snap_counter,u_d)
   call measure_divergence_ldf(total_divB, surf_divB, vol_divB,db_d)
   call write_divergence(t,total_divB, surf_divB, vol_divB)

   print '("cpu average Time = ",e10.3," seconds.")',(finish-start)/dble(iter)

end subroutine evolve_ldf

subroutine evolve(u,u_d,du_d,dudt_d,w_d)
  ! eat real nodal values, spit out nodal values
  USE ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: u_d, du_d, dudt_d, w_d
  ! internal variables
  real(kind=8)::t,dt,top,start,finish,time
  real(kind=8)::cmax, dx, dy, cs_max,v_xmax,v_ymax
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,dudt,w1,w2,w3,w4,delta_u,nodes
  real(kind=8)::total_divB, surf_divB, vol_divB, vm, total_divB_w
  real(kind=8):: cp2, ch, decay_coeff
  integer::iter, n, i, j, irk, rkt, ssp
  integer::snap_counter
  integer::var

  snap_counter = 0
  call writefields_new(snap_counter,du_d)
  snap_counter = 1
  call device_get_modes_from_nodes(u_d,du_d)
  !call device_compute_limiter(du_d)
  call writefields_new(snap_counter,du_d)

  t=0
  top=dtop
  iter=0
  ssp=m
  dx = 1./nx
  open(12, file=trim(folder)//"/divB.txt", status="replace")
  close(12)
  call measure_divergence(total_divB, surf_divB, vol_divB)
  call write_divergence(t,total_divB, surf_divB, vol_divB)
  print*,'entered evolve'

  if(ssp<4)then
     rkt= 4! ssp-1
     ssp = 4
  else if(ssp>=4) then
     rkt = 4
     ssp = 4
  end if

  dtop = 0.0005
  top = dtop
  call cpu_time(start)

  !do while(iter<10)
  do while(t < tend)
    ! Compute time step
    call device_compute_min_dt(dt)
    call device_compute_max_v(vm)

    if (corr) then
        if (corr_type == 'DED') then
          ch =  5.0*vm
        else if (corr_type == 'GUI') then
          ch =  (0.95/(2.*m-1.)*(1./nx)*(1./(2.*dt)))
        else if (corr_type == 'ZAN') then
          ch =  2.0*vm
        end if

        dt = min(dt,cfl/(2*m-1)/(2.*max(ch,vm)*nx))
        dt = min(dt,tend-t)

        if (corr_type == 'DED') then
          cp2 = 0.18*ch
          decay_coeff = (ch*ch/cp2)*dt
        else if (corr_type == 'GUI') then
          cp2 = 10.0*dt*ch*ch
          decay_coeff = (ch*ch/cp2)*dt
        else if (corr_type == 'ZAN') then
          decay_coeff = 0.8*ch/(dx/dt)
        end if
    end if

    do irk=0,rkt !RUNGE KUTTA SUBSTEPING
      call device_compute_update(irk,ssp,dt,t,vm)
    end do

    if (corr) then
      do irk=0,rkt
         call post_process_b(irk,ssp,dt,t,vm, ch)
      end do
    end if
    if (corr) then
      call parabolic_psi(dt,t,vm,dx, decay_coeff)
    end if

    if (corr) then
        t=t + dt
    else
        t = t + dt
    end if
    iter=iter+1
    !write(*,*)'time=',iter,t,dt

    if (t>=0.02) then
          dtop = 0.01
    end if

    if ((make_movie).and.(t>=top))then
       print*, dt, cfl/(2*m-1)/(2.*vm*vm*nx)
       snap_counter = snap_counter + 1
       call writefields_new(snap_counter,du_d)
       call measure_divergence(total_divB, surf_divB, vol_divB)
       call write_divergence(t,total_divB, surf_divB, vol_divB)
       call measure_divergence_wasilij(total_divB_w)
       top = top + dtop

      write(*,*)'time=',iter,t,dt
      write(*,*)'ch,cp,vm=',ch,cp2,vm

    end if
    if (iter==1) then
      call writefields_new(-1,du_d)
    end if



 end do
 snap_counter = snap_counter + 1
 call writefields_new(snap_counter,du_d)
 call measure_divergence(total_divB, surf_divB, vol_divB)
 call write_divergence(t,total_divB, surf_divB, vol_divB)
 call measure_divergence_wasilij(total_divB_w)

 call cpu_time(finish)

 print '("cpu average Time = ",e10.3," seconds.")',(finish-start)/dble(iter)
end subroutine evolve


subroutine read_params
  USE ISO_C_BINDING
  use parameters_dg_2d
  !character(LEN=80)::infile, info_file
  character(LEN=50)::arg
  character(LEN=12)::cninit,cnx,cm
  character(LEN=20)::ceta

  call getarg(1,arg)
  read(arg,*) nx

  call getarg(2,arg)
  read(arg,*) ny

  call getarg(3,arg)
  read(arg,*) m

  call getarg(4,arg)
  read(arg,*) ninit

  call getarg(5,arg)
  read(arg,*) corr_type

  call getarg(6,arg)
  read(arg,*) folder


  k=(m+3)/2

  write (cnx,"(I2)") nx

  folder = "runs/"//trim(folder)
end subroutine read_params
