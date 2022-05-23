program main
  USE ISO_C_BINDING
  use parameters_dg_2d
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u,w  
  !device pointers
  type (c_ptr) :: u_d, du_d,dudt_d, w_d, u_eq_d 
  type (c_ptr) :: x_d, y_d
  !Initialization of fields
  call get_coords()
  call get_initial_conditions(u,w)
  !call output_file(x,y,u,1,'initcond')
  call get_equilibrium_solution()
 
  !device allocation and memory copies 
  call devices()
  call setdevice(0)
  call gpu_allocation(nvar,nx,ny,m,k,boxlen_x,boxlen_y,cfl,eta,gamma)
  call gpu_set_pointers(u_d,du_d,dudt_d,w_d,u_eq_d,x_d,y_d &
            & ,x_quad,y_quad,w_x_quad,w_y_quad,x_gll,y_gll,w_x_gll,w_y_gll,sqrt_mod)
  call h2d(u,u_d,nvar*nx*ny*m*m)
  call h2d(w,w_d,nvar*nx*ny*m*m)
  call h2d(u_eq,u_eq_d,nvar*nx*ny*m*m)
  call h2d(x,x_d,nx*ny*m*m)
  call h2d(y,y_d,nx*ny*m*m)
  
  call evolve(u, u_d, du_d, dudt_d)

end program main

subroutine evolve(u,u_d,du_d,dudt_d)
  ! eat real nodal values, spit out nodal values
  USE ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: u_d, du_d, dudt_d
  ! internal variables
  real(kind=8)::t,dt,top,dtop
  real(kind=8)::cmax, dx, dy, cs_max,v_xmax,v_ymax
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u,dudt,w1,w2,w3,w4,delta_u,nodes
  integer::iter, n, i, j, irk, rkt, ssp
  integer::snap_counter
  integer::var
  character(len=10)::filename

  dx = boxlen_x/dble(nx)
  dy = boxlen_y/dble(ny)
  delta_u(:,:,:,:,:) = 0.0
  
  !call device_get_modes_from_nodes(u,u_d)
  !call d2h(u_d,u,nvar*nx*ny*m*m)
  !delta_u = u
  !call device_test_routine(u)
  !call d2h(u,delta_u,nvar*nx*ny*m*m)
  
  call device_get_modes_from_nodes(u_d,du_d)
  call d2h(du_d,u,nvar*nx*ny*m*m)
  call d2h(du_d,delta_u,nvar*nx*ny*m*m)
  call device_test_routine(du_d)
  call d2h(du_d,delta_u,nvar*nx*ny*m*m)


  write(*,*) maxval(abs(delta_u-u))   
  write(*,*) 'u'
  write(*,*) u(:,:,128,128,1)
  write(*,*) 'changed u'
  write(*,*) delta_u(:,:,128,128,1)
  write(*,*) u(:,:,128,128,2)
  write(*,*) delta_u(:,:,128,128,2)
  write(*,*) u(:,:,128,128,3)
  write(*,*) delta_u(:,:,128,128,3)
  write(*,*) u(:,:,128,128,4)
  write(*,*) delta_u(:,:,128,128,4)
  do i = 1,2
   do j = 1,2
    if (maxval(abs(delta_u(:,:,i,j,:) - u(:,:,i,j,:))) > 100) then
        write(*,*) 'delta u',delta_u(:,:,i,j,:)
        write(*,*) 'original u', u(:,:,i,j,:)
        write(*,*) i,' ',j
    end if 
   end do
  end do 
end subroutine evolve






