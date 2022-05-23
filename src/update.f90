
subroutine compute_update(delta_u,dudt)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::delta_u,dudt
  real(kind=8),dimension(1:nvar,1,1:m,1:nx+1,1:ny)::F
  real(kind=8),dimension(1:nvar,1:m,1,1:nx, 1:ny+1)::G
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::s, source_vol
  !===========================================================
  ! This routine computes the DG update for the input state u.
  !===========================================================
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nx+1,1:ny+1)::x_faces, y_faces
  real(kind=8),dimension(1:nvar,1:m,1:m)::u_quad,source_quad

  real(kind=8),dimension(1:nvar):: u_temp
  real(kind=8),dimension(1:nvar,2):: flux_temp
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u_delta_quad
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::flux_quad1
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::flux_quad2


  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::flux_vol1, flux_vol2
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1,1:m)::u_left,u_right
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1)::u_top, u_bottom
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1,2)::flux_top, flux_bottom

  real(kind=8),dimension(1:nvar)::flux_riemann,u_tmp
  real(kind=8),dimension(1:nvar,1:nx+1,1:ny+1)::flux_face,flux_face_eq
  !real(kind=8),dimension(1:nx+1,1:ny+1)::x_faces
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m,2)::flux_left,flux_right
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar,4):: edge

  integer::icell,i,j,iface,ileft,iright,ivar, node, jface
  integer::intnode,jntnode,jcell, edge_num, one
  real(kind=8)::legendre,legendre_prime
  real(kind=8)::chsi_left=-1,chsi_right=+1
  real(kind=8)::chsi_bottom=-1,chsi_top=+1
  real(kind=8)::dx,dy
  real(kind=8)::cmax,oneoverdx,c_left,c_right,oneoverdy
  real(kind=8)::x_right,x_left
  real(kind=8),dimension(1:nvar,1:m):: u_delta_r, u_delta_l, u_delta_t, u_delta_b
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m)::u_delta_left,u_delta_right,u_delta_top,u_delta_bottom
  real(kind=8),dimension(1:nvar,1:(nx+1))::u_face_eq, w_face_eq

  ! do nothing

end subroutine compute_update

subroutine apply_limiter(u)
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u

  if(use_limiter) then
    if (limiter_type == '1OR') then
      call compute_limiter(u)
    else if (limiter_type == 'HIO') then
      call high_order_limiter(u)
    else if (limiter_type=='LOW') then
      call limiter_low_order(u)

    else if (limiter_type=='POS') then
      call limiter_positivity(u)

    else if (limiter_type=='ROS') then
      call limiter_rossmanith(u)

    end if

  end if

end subroutine apply_limiter
