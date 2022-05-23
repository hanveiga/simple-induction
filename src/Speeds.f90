subroutine compute_max_speed(u, cs_max, v_xmax, v_ymax, speed_max)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::u
  integer::icell, jcell, j,i
  real(kind=8)::speed, cs, v_x, v_y, cs_max, v_xmax, v_ymax, speed_max
  real(kind=8),dimension(1:nvar)::w
  ! Compute max sound speed
  speed_max=0.0
  do icell=1,nx
    do jcell = 1,ny
      do i=1,m
        do j=1,m
          call compute_speed(u(j,i,jcell,icell,1:nvar),cs,v_x,v_y,speed)
          if (speed >= speed_max) then
            speed_max=MAX(speed_max,speed)
            v_xmax = v_x
            v_ymax = v_y
            cs_max = cs
            !cs_max = sqrt(1.4)
            call compute_primitive(u(j,i,jcell,icell,1:nvar),w,1,1,1,1)
          end if
        end do
      end do
    end do
  end do
end subroutine compute_max_speed


subroutine compute_speed(u,cs,v_x,v_y,speed)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:nvar)::u
  real(kind=8)::speed
  real(kind=8),dimension(1:nvar)::w
  real(kind=8)::cs, v_x, v_y
  ! Compute primitive variables
  call compute_primitive(u,w,1,1,1,1)
  ! Compute sound speed

  cs=sqrt(gamma*max(w(4),1d-10)/max(w(1),1d-10))
  v_x = w(2)
  v_y = w(3)
  speed=sqrt(w(2)**2+w(3)**2)+cs
end subroutine compute_speed
