
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

subroutine get_coords_general_quad(offset_x,offset_y, x_gen, y_gen)
  use parameters_dg_2d
  implicit none

  !internal vars
  real(kind=8)::dx, dy, offset_x, offset_y
  integer::i,j, nj, ni
  real(kind=8),dimension(1:m+1)::x_quad_gen, w_x_quad_gen
  real(kind=8),dimension(1:m+1)::y_quad_gen, w_y_quad_gen

  real(kind=8),dimension(1:nx,1:ny,1:m+1,1:m+1)::x_gen
  real(kind=8),dimension(1:nx,1:ny,1:m+1,1:m+1)::y_gen

  ! get quadrature
  call gl_quadrature(x_quad_gen,w_x_quad_gen,m+1)
  call gl_quadrature(y_quad_gen,w_y_quad_gen,m+1)

  !do i=1,m+1
  !  sqrt_mod(i)=sqrt((2.0*dble(i-1)+1.0))
  !end do
  dx = boxlen_x/dble(nx)
  dy = boxlen_y/dble(ny)

  do i=1,nx
    do j=1,ny
      do ni =1,m+1
        do nj = 1,m+1
          x_gen(i,j,ni,nj) = (i-0.5)*dx + dx/2.0*x_quad_gen(ni) - offset_x !- 0.5*boxlen_x
          y_gen(i,j,ni,nj) = (j-0.5)*dy + dy/2.0*y_quad_gen(nj) - offset_y !- 0.5*boxlen_y
        end do
      end do
    end do
  end do

end subroutine get_coords_general_quad

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
                w(icell,jcell,inti,intj,1)  =  2.0*50.*(y(icell,jcell,inti,intj)-0.5*boxlen_y)*exp(-50.*r**2)**2.  !-A_0*(yb(inti,intj,icell,jcell)-0.5*boxlen_y)/(r+0.0001)
                w(icell,jcell,inti,intj,2)  = -2.0*50.*(x(icell,jcell,inti,intj)-0.5*boxlen_x)*exp(-50.*r**2)**2.! A_0*(xb(inti,intj,icell,jcell)-0.5*boxlen_x)/(r+0.0001)
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
      tend = 0.01
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


subroutine initialise_magnetic_potential(A_pot)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:nx,1:ny,1:m+1,1:m+1)::A_pot
  integer::i,j,inti,intj, jcell, icell, n

  ! internal variables
  real(kind=8)::r0, omega, cs, grav, Ms, eps, h, vt, vk, R_0, A_0, rc
  real(kind=8)::x_dash, y_dash, r, rho_d, delta_r, x_center, y_center, cutoff
  integer::deg_x, deg_y

  real(kind=8),dimension(1:nx,1:ny,1:m+1,1:m+1)::x_gen_coords, y_gen_coords
  real(kind=8)::xloc,yloc

  !deg_x = m + 1
  !deg_y = m + 1

  select case (ninit)

  case(8) ! discontinuous magnetic potential
    if (ignore) then
      boxlen_x = 1.0
      boxlen_y = 1.0
      gamma = 0.0
      tend = 0.01
    end if

    call get_coords_general_quad(0.0,0.0,x_gen_coords,y_gen_coords)

    R_0 = 0.3
    A_0 = 0.001
    do icell = 1,nx
      do jcell = 1,ny
        do inti = 1,m+1
          do intj = 1,m+1
              r = sqrt((x_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_x)**2.0 + &
                  & (y_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_y)**2.0 )
              xloc = x_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_x
              yloc = y_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_y
              if (r<R_0) then
                write(*,*) 'nonzero'
                 A_pot(icell,jcell,inti,intj) = A_0*(R-r)
              else
                 A_pot(icell,jcell,inti,intj) = 0.0
              end if
              !A_pot(icell,jcell,inti,intj) = cos(xloc*yloc)

          end do
          end do

      end do
  end do
  case(9)
    if (ignore) then
        boxlen_x = 1.0
        boxlen_y = 1.0
        gamma = 0.0
        tend = 0.01
      end if

      call get_coords_general_quad(0.0,0.0,x_gen_coords,y_gen_coords)

      R_0 = 0.3
      A_0 = 0.001
      do icell = 1,nx
        do jcell = 1,ny
          do inti = 1,m+1
            do intj = 1,m+1
                r = sqrt((x_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_x)**2.0 + &
                    & (y_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_y)**2.0 )
                xloc = x_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_x
                yloc = y_gen_coords(icell,jcell,inti,intj)-0.5*boxlen_y
                A_pot(icell,jcell,inti,intj)  =  exp(-60.*r**2)
            end do
            end do

        end do
    end do


  end select

end subroutine initialise_magnetic_potential


subroutine get_B_from_A(A, u,w)
  use parameters_dg_2d
  implicit none
! compute modes for A (m+1) modes
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u,w
  real(kind=8),dimension(1:nx,1:ny,1:m+1,1:m+1)::A, A_modes
  real(kind=8),dimension(1:nx,1:ny,1:m+1,1:m+1,2)::B_nodes
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,2):: B_modes

  real(kind=8),dimension(1:m+1)::x_quad_gen, w_x_quad_gen
  real(kind=8),dimension(1:m+1)::y_quad_gen, w_y_quad_gen

  real(kind=8)::legendre, legendre_prime

  integer::i,j,icell,jcell,xquad,yquad,intnode,jntnode
  ! get quadrature
  call gl_quadrature(x_quad_gen,w_x_quad_gen,m+1)
  call gl_quadrature(y_quad_gen,w_y_quad_gen,m+1)

  do icell=1,nx
    do jcell=1,ny
      do i=1,m+1
        do j=1,m+1
          ! Loop over quadrature points
          do xquad=1,m+1 ! for more general use n_x_quad...
            do yquad=1,m+1
              ! Quadrature point in physical space
              ! Perform integration using GL quadrature
              A_modes(icell,jcell,i,j)=A_modes(icell,jcell,i,j)+ &
              & 0.25*A(icell,jcell,xquad,yquad)* &
              & legendre(x_quad_gen(xquad),i-1)* &
              & legendre(y_quad_gen(yquad),j-1)* &
              & w_x_quad_gen(xquad) * &
              & w_y_quad_gen(yquad)
            end do
          end do
        end do
      end do
    end do
  end do

write(*,*) "A modes"
write(*,*) A_modes

! take derivative of A to get B_x, B_y (nodes)

  do icell=1,nx
      do jcell = 1,ny
        do i=1,m+1
          do j=1,m+1
            do intnode = 1,m+1
              do jntnode = 1,m+1
                ! Loop over quadrature points
                B_nodes(icell,jcell,i,j,1) = B_nodes(icell,jcell,i,j,1) -&
                & A_modes(icell,jcell,intnode,jntnode)*&
                &legendre(x_quad_gen(i),intnode-1)*&
                &legendre_prime(y_quad_gen(j),jntnode-1)

                B_nodes(icell,jcell,i,j,2) = B_nodes(icell,jcell,i,j,2) +&
                & A_modes(icell,jcell,intnode,jntnode)*&
                &legendre_prime(x_quad_gen(i),intnode-1)*&
                &legendre(y_quad_gen(j),jntnode-1)

              end do
            end do
          end do
        end do
      end do
    end do


write(*,*) "B nodes"
write(*,*) B_nodes

! integrate nodes to get modes of B_x (up to m)
  do icell=1,nx
    do jcell=1,ny
      do i=1,m
        do j=1,m
          do xquad=1,m+1 
            do yquad=1,m+1
              B_modes(icell,jcell,i,j,1)=B_modes(icell,jcell,i,j,1)+ &
              & 0.25*B_nodes(icell,jcell,xquad,yquad,1)* &
              & legendre(x_quad_gen(xquad),i-1)* &
              & legendre(y_quad_gen(yquad),j-1)* &
              & w_x_quad_gen(xquad) * &
              & w_y_quad_gen(yquad)

              B_modes(icell,jcell,i,j,2)=B_modes(icell,jcell,i,j,2)+ &
              & 0.25*B_nodes(icell,jcell,xquad,yquad,2)* &
              & legendre(x_quad_gen(xquad),i-1)* &
              & legendre(y_quad_gen(yquad),j-1)* &
              & w_x_quad_gen(xquad) * &
              & w_y_quad_gen(yquad)
            end do
          end do
        end do
      end do
    end do
  end do


write(*,*) "B modes"
write(*,*) B_modes

  do icell=1,nx
      do jcell = 1,ny
        do i=1,m
          do j=1,m
            do intnode = 1,m
              do jntnode = 1,m
                ! Loop over quadrature points
                w(icell,jcell,i,j,1) = w(icell,jcell,i,j,1) - &
                & B_modes(icell,jcell,intnode,jntnode,1)*&
                &legendre(x_quad(i),intnode-1)*&
                &legendre_prime(y_quad(j),jntnode-1)

                w(icell,jcell,i,j,2) = w(icell,jcell,i,j,2) +&
                & B_modes(icell,jcell,intnode,jntnode,2)*&
                &legendre_prime(x_quad(i),intnode-1)*&
                &legendre(y_quad(j),jntnode-1)

                w(icell,jcell,i,j,3) = 0.0
                w(icell,jcell,i,j,4) = 0.0
              end do
            end do
          end do
        end do
      end do
    end do

call compute_conservative(w,u,nx,ny,m)
end subroutine