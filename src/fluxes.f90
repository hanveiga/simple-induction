subroutine get_boundary_conditions(index, dim)
  use parameters_dg_2d

  integer::index
  integer::dim

  ! if direction = 1 then its x direction, etc.

  if (dim == 1) then
    index = index

    if (bc == 1) then !periodic
      if (index == 0) then
        index = nx
      else if (index == nx+1) then
        index = 1
      end if

    else if ((bc == 2).or.(bc==3)) then !transmissive or reflective
      if (index == 0) then
        index = 1
      else if (index == nx+1) then
        index = nx
      end if

    end if

  else if(dim == 2) then

    if (bc == 1) then !periodic
      if (index == 0) then
        index = ny
      else if (index == ny+1) then
        index = 1
      end if

    else if ((bc == 2).or.(bc==3)) then !transmissive
      if (index == 0) then
        index = 1
      else if (index == ny+1) then
        index = ny
      end if

    end if

  end if

end subroutine get_boundary_conditions

subroutine compute_flux(u,flux1, flux2, size_x,size_y,order)
  use parameters_dg_2d

  ! do nothing
  
end subroutine compute_flux

subroutine compute_flux_int(u,flux)
  use parameters_dg_2d

! do nothing

end subroutine compute_flux_int


subroutine compute_llflux(uleft,uright, f_left,f_right, fgdnv, flag)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:nvar)::uleft,uright, f_left, f_right
  real(kind=8),dimension(1:nvar)::fgdnv
  real(kind=8)::cleft,cs,cmax,speed_left,speed_right,v_x,v_y
  real(kind=8),dimension(1:nvar)::fleft,fright
  integer::flag
  ! Maximum wave speed
  call compute_speed(uleft,cs,v_x,v_y,speed_left)
  call compute_speed(uright,cs,v_x,v_y,speed_right)

  !subroutine compute_speed(u,cs,v_x,v_y,speed)
  cmax=max(speed_left,speed_right)

  ! Compute Godunox flux
  fgdnv=0.5*(f_right+f_left)+0.5*cmax*(uleft-uright)
end subroutine compute_llflux
!-----
