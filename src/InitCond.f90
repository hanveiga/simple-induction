!---------
subroutine get_coords(offset_x,offset_y)
  use parameters_dg_2d
  implicit none

  !internal vars
  real(kind=8)::dx, dy, offset_x, offset_y
  integer::i,j, nj, ni

  ! get quadrature
  call gl_quadrature(x_quad,w_x_quad,m)
  call gl_quadrature(y_quad,w_y_quad,m)
  call gll_quadrature(x_gll,w_x_gll,k)
  call gll_quadrature(y_gll,w_y_gll,k)

  do i=1,m
    sqrt_mod(i)=sqrt((2.0*dble(i-1)+1.0))
  end do
  dx = boxlen_x/dble(nx)
  dy = boxlen_y/dble(ny)
  do i=1,nx
    do j=1,ny
      do ni =1,m
        do nj = 1,m
          x(i,j,ni,nj) = (i-0.5)*dx + dx/2.0*x_quad(ni) - offset_x !- 0.5*boxlen_x
          y(i,j,ni,nj) = (j-0.5)*dy + dy/2.0*y_quad(nj) - offset_y !- 0.5*boxlen_y
        end do
      end do
    xc(i,j) = (i-0.5)*dx - offset_x
    yc(i,j) = (j-0.5)*dx - offset_y
    end do
  end do


  sqrts_div(1)=sqrt(dble(2.0))
  sqrts_div(2)=sqrt(dble(3.0))
end subroutine get_coords

!-------
subroutine initialisation(u,w)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,w
  integer::i,j,inti,intj, jcell, icell, n

  ! internal variables
  real(kind=8)::rho_0, p_0, g
  real(kind=8)::dpi=acos(-1d0)
  real(kind=8)::beta
  real(kind=8)::r0, omega, cs, grav, Ms, eps, h, vt, vk, R_0, A_0, rc
  real(kind=8)::x_dash, y_dash, r, rho_d, delta_r, x_center, y_center, cutoff

  select case (ninit)
  case(1) !Linear advection of pulse
    if (ignore) then
      boxlen_x = 2.*dpi
      boxlen_y = 2.*dpi
      gamma = 2.0
      tend = 7.0
    end if
    call get_coords(0.0,0.0)

      w(:,:,:,:,1) = 2. + sin(x + y)
      w(:,:,:,:,2) = 1.0
      w(:,:,:,:,3) = 0.0
   case(2) ! totally smooth pulse
    if (ignore) then
      boxlen_x = 1.0
      boxlen_y = 1.0
      gamma = 0.0
      tend = 1.0
    end if
    call get_coords(0.0,0.0)

      w(:,:,:,:,1) = -2*dpi*sin(2.0*dpi*x(:,:,:,:))*cos(2*dpi*y(:,:,:,:))
      w(:,:,:,:,2) = 2*dpi*cos(2.0*dpi*x(:,:,:,:))*sin(2*dpi*y(:,:,:,:))
      w(:,:,:,:,3) = 0.0
      w(:,:,:,:,4) = 0.0
  case(9)
      if (ignore) then
        boxlen_x = 1.0
        boxlen_y = 1.0
        gamma = 0.0
        tend = 2.0
      end if

      call get_coords(0.0,0.0)
      do icell = 1,nx
        do jcell = 1,ny
          rc = sqrt((x(icell,jcell,1,1)-0.5*boxlen_x)**2 + (y(icell,jcell,1,1)-0.5*boxlen_y)**2 )
          do inti = 1,m
            do intj = 1,m
                r = sqrt((x(icell,jcell,inti,intj)-0.5*boxlen_x)**2 + (y(icell,jcell,inti,intj)-0.5*boxlen_y)**2 )

                !if (rc<R_0) then
                w(icell,jcell,inti,intj,1)  =  2.0*50.*(y(icell,jcell,inti,intj)-0.5*boxlen_y)*exp(-50.*r**2)  !-A_0*(yb(inti,intj,icell,jcell)-0.5*boxlen_y)/(r+0.0001)
                w(icell,jcell,inti,intj,2)  = -2.0*50.*(x(icell,jcell,inti,intj)-0.5*boxlen_x)*exp(-50.*r**2)! A_0*(xb(inti,intj,icell,jcell)-0.5*boxlen_x)/(r+0.0001)
                !else
                !  w(icell,jcell,inti,intj,1)  = 0.0
                !  w(icell,jcell,inti,intj,2)  = 0.0
                !end if

                w(icell,jcell,inti,intj,3)  = 0.
                w(icell,jcell,inti,intj,4)  = 0.
            end do
            end do
        end do
    end do
  case(8) ! discontinuous
    if (ignore) then
      boxlen_x = 1.0
      boxlen_y = 1.0
      gamma = 0.0
      tend = 2.0
    end if

    call get_coords(0.0,0.0)
    R_0 = 0.3
    A_0 = 0.001
    do icell = 1,nx
      do jcell = 1,ny
        !rc = sqrt((x(icell,jcell,1,1)-0.5*boxlen_x)**2 + (y(icell,jcell,1,1)-0.5*boxlen_y)**2 )
        do inti = 1,m
          do intj = 1,m
              r = sqrt((x(icell,jcell,inti,intj)-0.5*boxlen_x)**2.0 + (y(icell,jcell,inti,intj)-0.5*boxlen_y)**2.0 )

              if (r<R_0) then
              w(icell,jcell,inti,intj,1)  =  -A_0*(y(icell,jcell,inti,intj)-0.5*boxlen_y)/(r)
              w(icell,jcell,inti,intj,2)  =  A_0*(x(icell,jcell,inti,intj)-0.5*boxlen_x)/(r)
              else
                w(icell,jcell,inti,intj,1)  = 0.0
                w(icell,jcell,inti,intj,2)  = 0.0
              end if

              w(icell,jcell,inti,intj,3)  = 0.
              w(icell,jcell,inti,intj,4)  = 0.
          end do
          end do

      end do
  end do
case(5) ! discontinuous
  if (ignore) then
    boxlen_x = 2.0
    boxlen_y = 2.0
    gamma = 0.0
    tend = 0.50
  end if

  call get_coords(0.0,0.0)
  R_0 = 0.3
  A_0 = 0.001
  do icell = 1,nx
    do jcell = 1,ny
      rc = sqrt((x(icell,jcell,1,1)-0.25*boxlen_x)**2 + (y(icell,jcell,1,1)-0.25*boxlen_y)**2 )
      do inti = 1,m
        do intj = 1,m
            r = sqrt((x(icell,jcell,inti,intj)-0.25*boxlen_x)**2 + (y(icell,jcell,inti,intj)-0.25*boxlen_y)**2 )

            if (rc<R_0) then
            w(icell,jcell,inti,intj,1)  =  -A_0*(y(icell,jcell,inti,intj)-0.25*boxlen_y)/r
            w(icell,jcell,inti,intj,2)  =  A_0*(x(icell,jcell,inti,intj)-0.25*boxlen_x)/r
            else
              w(icell,jcell,inti,intj,1)  = 0.0
              w(icell,jcell,inti,intj,2)  = 0.0
            end if

            w(icell,jcell,inti,intj,3)  = 0.
            w(icell,jcell,inti,intj,4)  = 0.
        end do
        end do

    end do
end do
end select
call compute_conservative(w,u,nx,ny,m)

end subroutine initialisation
