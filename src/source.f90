subroutine get_source(u,s)
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u,s,w
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,2)::grad_p
  
  call compute_primitive(u,w,nx,ny,m)
  call grad_phi(u,grad_p)

  s(:,:,:,:,1) = 0.
  s(:,:,:,:,2) = w(:,:,:,:,1)*grad_p(:,:,:,:,1)
  s(:,:,:,:,3) = w(:,:,:,:,1)*grad_p(:,:,:,:,2)
  s(:,:,:,:,4) = w(:,:,:,:,1)*(w(:,:,:,:,2)*grad_p(:,:,:,:,1)&
  &+w(:,:,:,:,3)*grad_p(:,:,:,:,2))

end subroutine get_source
!--------

subroutine grad_phi(u, grad_p)
  use parameters_dg_2d
  implicit none
  ! this could be a function by the way
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,2)::grad_p
  
  real(kind=8)::delta_r,x_center,y_center
  real(kind=8)::x_dash,y_dash, r, epsilon

  integer::i,j,icell,jcell

  epsilon = 0.25
  delta_r = 0.1
  x_center = 3.
  y_center = 3.

  do icell = 1,nx
    do jcell = 1,ny
      do i = 1,m
        do j = 1,m
          x_dash = x(j,i,jcell,icell) - x_center
          y_dash = y(j,i,jcell,icell) - y_center
          r = sqrt(x_dash**2 + y_dash**2)

          if (r > 0.5-0.5*delta_r) then
            grad_p(j,i,jcell,icell,1) = -(x_dash)/(r**3)
            grad_p(j,i,jcell,icell,2) = -(y_dash)/(r**3)
          else if (r <= 0.5-0.5*delta_r) then
            grad_p(j,i,jcell,icell,1) = -(x_dash)/(r*(r**2+epsilon**2))
            grad_p(j,i,jcell,icell,2) = -(y_dash)/(r*(r**2+epsilon**2))
          end if

        end do
      end do
    end do
  end do

end subroutine


 
