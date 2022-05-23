
subroutine get_modes_from_nodes(nodes, modes)
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::modes,nodes

  ! internal variables
  integer::icell, jcell, i, j, xquad, yquad
  real(kind=8)::legendre

  modes(:,:,:,:,:) = 0.0

  do icell=1,nx
    do jcell=1,ny
      do i=1,m
        do j=1,m
          ! Loop over quadrature points
          do xquad=1,m ! for more general use n_x_quad...
            do yquad=1,m
              ! Quadrature point in physical space
              ! Perform integration using GL quadrature
              modes(j,i,jcell,icell,1:nvar)=modes(j,i,jcell,icell,1:nvar)+ &
              & 0.25*nodes(yquad,xquad,jcell,icell,1:nvar)* &
              & legendre(x_quad(xquad),i-1)* &
              & legendre(y_quad(yquad),j-1)* &
              & w_x_quad(xquad) * &
              & w_y_quad(yquad)
            end do
          end do
        end do
      end do
    end do
  end do
end subroutine get_modes_from_nodes

subroutine get_nodes_from_modes(modes,nodes)
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::modes,nodes

  ! internal variables
  integer::icell, jcell, i, j, intnode, jntnode, ivar
  real::xquad,yquad
  real(kind=8)::legendre

  nodes(:,:,:,:,:) = 0.0

  do ivar = 1,nvar
    do icell=1,nx
      do jcell = 1,ny
        do i=1,m
          do j=1,m
            do intnode = 1,m
              do jntnode = 1,m
                ! Loop over quadrature points
                nodes(j,i,jcell,icell,ivar) = nodes(j,i,jcell,icell,ivar) +&
                & modes(jntnode,intnode,jcell,icell,ivar)*&
                &legendre(x_quad(i),intnode-1)*&
                &legendre(y_quad(j),jntnode-1)
              end do
            end do
          end do
        end do
      end do
    end do
  end do

end subroutine get_nodes_from_modes

subroutine compute_primitive(u,w,size_x,size_y,order)
  use parameters_dg_2d
  implicit none
  integer::size_x,size_y,order
  real(kind=8),dimension(1:order,1:order,1:size_y,1:size_x,1:nvar)::u,w
  !Compute primitive variables
  w(:,:,:,:,1) = u(:,:,:,:,1)
  w(:,:,:,:,2) = u(:,:,:,:,2)
  w(:,:,:,:,3) = u(:,:,:,:,3)
  w(:,:,:,:,4) = u(:,:,:,:,4)
end subroutine compute_primitive

subroutine compute_conservative(ww,u,size_x,size_y,order)
  use parameters_dg_2d
  implicit none
  integer::size_x,size_y,order
  real(kind=8),dimension(1:order,1:order,1:size_y,1:size_x,1:nvar)::u,ww
  ! Compute primitive variables
  u(:,:,:,:,1)= ww(:,:,:,:,1)
  u(:,:,:,:,2)= ww(:,:,:,:,2)
  u(:,:,:,:,3)= ww(:,:,:,:,3)
  u(:,:,:,:,4)= ww(:,:,:,:,4)
  !ww(3,:,:)/(gamma-1.0)+0.5*ww(1,:,:)*ww(2,:,:)**2
end subroutine compute_conservative


!!!!!!!!
subroutine get_modes_from_nodes_cpu(nodes, modes)
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::nodes
  real(kind=8),dimension(1:nx,1:ny,1:(m)*(m+3)/2)::modes

  ! internal variables
  integer::icell, jcell, i, j, xquad, yquad
  real(kind=8)::div_b_basis_tri

  modes(:,:,:) = 0.0

  do icell=1,nx
    do jcell=1,ny
      do i=1, (m)*(m+3)/2
          ! Loop over quadrature points
          do xquad=1,m ! for more general use n_x_quad...
            do yquad=1,m
              ! Quadrature point in physical space
              ! Perform integration using GL quadrature
              modes(icell,jcell,i)=modes(icell,jcell,i)+ &
              & 0.25*(nodes(icell,jcell,xquad,yquad,1)* &
              & div_b_basis_tri( x_quad(xquad),  y_quad(yquad),  i-1,  0) &
              & + nodes(icell,jcell,xquad,yquad,2)* &
              & div_b_basis_tri( x_quad(xquad),  y_quad(yquad),  i-1,  1) )* &
              & w_x_quad(xquad) * &
              & w_y_quad(yquad)
            end do
          end do
      end do
    end do
  end do
end subroutine get_modes_from_nodes_cpu

subroutine get_nodes_from_modes_cpu(modes,nodes)
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::nodes
  real(kind=8),dimension(1:nx,1:ny,1:(m)*(m+3)/2)::modes
  ! internal variables
  integer::icell, jcell, i, j, intnode, jntnode, ivar
  real::xquad,yquad
  real(kind=8)::div_b_basis_tri

  nodes(:,:,:,:,:) = 0.0

  do icell=1,nx
      do jcell = 1,ny
        do i=1,m
          do j=1,m
            do intnode = 1,(m)*(m+3)/2
                ! Loop over quadrature points
                nodes(icell,jcell,i,j,1) = nodes(icell,jcell,i,j,1) +&
                & modes(icell,jcell,intnode)*&
                & div_b_basis_tri( x_quad(i),  y_quad(j),  intnode-1,  0)

                nodes(icell,jcell,i,j,2) = nodes(icell,jcell,i,j,2) +&
                & modes(icell,jcell,intnode)*&
                & div_b_basis_tri( x_quad(i),  y_quad(j),  intnode-1,  1)
            end do
          end do
        end do
      end do
  end do

end subroutine get_nodes_from_modes_cpu
