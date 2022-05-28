subroutine measure_divergence(total_divB, surf_divB, vol_divB)
  use parameters_dg_2d
  implicit none

  integer::icell,jcell, inti, intj, ni,nj
  integer::im,ip,it,ib
  real(kind=8)::total_divB, surf_divB, vol_divB
  real(kind=8)::div_vol, div_surf, dx, dy, divx, divy, legendre_prime, legendre
  real(kind=8)::plus, minus, val
  real(kind=8)::basis_deriv, basis
  real(kind=8),dimension(4)::fc

  div_surf = 0.0
  div_vol = 0.0
  divx = 0.0
  divy = 0.0

  dx = 1.0/nx
  dy = 1.0/ny
  plus = 1.0
  minus = -1.0
  total_divB = 0
  surf_divB = 0
  vol_divB = 0
  fc = 0
  divx = 0
  divy = 0
  ! volume part
  do icell = 1,nx
    do jcell = 1,ny

      im = icell-1
      ip = icell+1
      ib = jcell-1
      it = jcell+1

      if (icell == 1) then
          im = nx
      end if

      if (icell == nx) then
          ip = 1
      end if

      if (jcell == ny) then
          it = 1
      end if

      if (jcell == 1) then
          ib = ny
      end if

      do inti = 1, m
        do intj = 1, m
          do ni = 1, m
            do nj = 1,m
              divx = divx +&
              &0.5*(currentsol(icell,jcell,ni,nj,1)*legendre_prime(x_quad(inti),ni-1)*legendre(y_quad(intj),nj-1)/dx&
              & + currentsol(icell,jcell,ni,nj,2)*legendre_prime(y_quad(intj),nj-1)*legendre(x_quad(inti),ni-1)/dy)&
              &*w_x_quad(intj)*w_y_quad(inti)*dx*dy
              end do
            end do
          div_vol = div_vol + abs(divx)
          divx = 0.0
        end do
      end do

      do inti = 1, m
        do ni = 1, m
          do nj = 1,m
            fc(1) = fc(1) + ((currentsol(icell,jcell,ni,nj,1)*legendre(plus,ni-1)*legendre(x_quad(inti),nj-1)) &
                    &-(currentsol(ip,jcell,ni,nj,1)*legendre(minus,ni-1)*legendre(x_quad(inti),nj-1)))&
                    &*w_x_quad(inti)/2.*dy
            fc(2) = fc(2) +&
                       &(currentsol(icell,jcell,ni,nj,1)*legendre(minus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-currentsol(im,jcell,ni,nj,1)*legendre(plus,ni-1)*legendre(x_quad(inti),nj-1))&
                       &*w_x_quad(inti)/2.*dy


            fc(3) = fc(3) + (currentsol(icell,jcell,nj,ni,2)*legendre(plus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-currentsol(icell,it,nj,ni,2)*legendre(minus,ni-1)*legendre(x_quad(inti),nj-1))&
                       &*w_x_quad(inti)/2.*dx


            fc(4) = fc(4) + (currentsol(icell,jcell,nj,ni,2)*legendre(minus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-currentsol(icell,ib,nj,ni,2)*legendre(plus,ni-1)*legendre(x_quad(inti),nj-1))&
                       &*w_x_quad(inti)/2.*dx
          end do
        end do
        !write(*,*) fc
        div_surf = div_surf + sum(abs(fc))
        fc = 0.0
      end do

    end do
  end do
  print*,'divergence', div_vol+div_surf
  total_divB = div_vol+div_surf
  surf_divB = div_surf
  vol_divB = div_vol
end subroutine


subroutine measure_divergence_ldf(total_divB, surf_divB, vol_divB,b_d)
  USE ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: b_d
  real(kind=8),dimension(1:nx,1:ny,1:(m)*(m+3)/2)::bmodes
  integer::icell,jcell, inti, intj, ni,nj
  integer::im,ip,it,ib
  real(kind=8)::total_divB, surf_divB, vol_divB
  real(kind=8)::div_vol, div_surf, dx, dy, divx, divy, legendre_prime, legendre
  real(kind=8)::plus, minus
  real(kind=8)::ldf_div_basis_tri_prime, div_b_basis_tri, val
  real(kind=8),dimension(4)::fc


  call d2h(b_d,bmodes,nx*ny*(m)*(m+3)/2)
  write(*,*) size(bmodes)
  div_surf = 0.0
  div_vol = 0.0
  divx = 0.0
  divy = 0.0

  dx = 1.0/nx
  dy = 1.0/ny
  plus = 1.0
  minus = -1.0
  total_divB = 0
  surf_divB = 0
  vol_divB = 0
  fc = 0
  divx = 0
  divy = 0
  ! volume part
  do icell = 1,nx
    do jcell = 1,ny

      im = icell-1
      ip = icell+1
      ib = jcell-1
      it = jcell+1

      if (icell == 1) then
          im = nx
      end if

      if (icell == nx) then
          ip = 1
      end if

      if (jcell == ny) then
          it = 1
      end if

      if (jcell == 1) then
          ib = ny
      end if

      do inti = 1, m
        do intj = 1, m
          !write(*,*) bmodes(icell,jcell,:)
          do ni = 1, (m)*(m+3)/2
           !  do nj = 1,m+1
              !write(*,*) ni
              divx = divx +&
              &0.5*bmodes(icell,jcell,ni)*(ldf_div_basis_tri_prime(x_quad(inti),y_quad(intj),ni-1, 0, 0)/dx&
              &+ldf_div_basis_tri_prime(x_quad(inti),y_quad(intj),ni-1,1, 1)/dy)&
              &*w_x_quad(intj)*w_y_quad(inti)*dx*dy
              end do
          !  end do
          div_vol = div_vol + abs(divx)
          divx = 0.0

          !write(*,*) 'div vol=',div_vol
        end do
      end do

      do inti = 1, m
        do ni = 1, (m)*(m+3)/2
          !do nj = 1,m+1
            !write(*,*) bmodes(icell,jcell,ni),bmodes(ip,jcell,ni)
            !val = basis(plus,x_quad(inti),ni-1, 0, 0, 0)
            !write(*,*) val
            !val = basis(minus,x_quad(inti),ni-1, 0 ,0, 0)
            !write(*,*) val
            fc(1) = fc(1) + &
                       &(bmodes(icell,jcell,ni)*div_b_basis_tri(plus,x_quad(inti),ni-1, 0)& !legendre(plus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-bmodes(ip,jcell,ni)*div_b_basis_tri(minus,x_quad(inti),ni-1, 0))&!legendre(minus,ni-1)*legendre(x_quad(inti),nj-1))*&
                       &*w_x_quad(inti)/2.*dy

            fc(2) = fc(2) +&
                       &(bmodes(icell,jcell,ni)*div_b_basis_tri(minus,x_quad(inti),ni-1, 0)&!legendre(minus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-bmodes(im,jcell,ni)*div_b_basis_tri(plus, x_quad(inti), ni-1, 0))&!legendre(plus,ni-1)*legendre(x_quad(inti),nj-1))&
                       &*w_x_quad(inti)/2.*dy


            fc(3) = fc(3) + (bmodes(icell,jcell,ni)*div_b_basis_tri(x_quad(inti),plus,ni-1, 1) &!legendre(plus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-bmodes(icell,it,ni)*div_b_basis_tri(x_quad(inti),minus,ni-1, 1))&!legendre(minus,ni-1)*legendre(x_quad(inti),nj-1))&
                       &*w_x_quad(inti)/2.*dx


            fc(4) = fc(4) + (bmodes(icell,jcell,ni)*div_b_basis_tri(x_quad(inti),minus,ni-1, 1)&!legendre(minus,ni-1)*legendre(x_quad(inti),nj-1)&
                       &-bmodes(icell,ib,ni)*div_b_basis_tri(x_quad(inti),plus,ni-1, 1))&!legendre(plus,ni-1)*legendre(x_quad(inti),nj-1))&
                       &*w_x_quad(inti)/2.*dx
          !end do
        end do
        div_surf = div_surf + sum(abs(fc))
        fc = 0.0
      end do
    end do
  end do
  print*,'divergence', div_vol+div_surf
  total_divB = div_vol+div_surf
  surf_divB = div_surf
  vol_divB = div_vol
end subroutine

subroutine write_divergence(t,total_divB, surf_divB, vol_divB)
  use parameters_dg_2d
  implicit none
  real(kind=8)::t, total_divB, surf_divB, vol_divB
  character(len=10)::filename

  logical :: exist

  write(filename,'(a)') 'divB.txt'

  !inquire(file=filename, exist=exist)
  !if (exist) then
  open(12, file=trim(folder)//"/"//filename, position="append", action="write")
  !else
  ! open(12, file=filename//".txt", status="new", action="write")
  !end if
  write(12, *) t, total_divB, surf_divB, vol_divB
  close(12)

end subroutine

subroutine project_b(device)
    use ISO_C_BINDING
    use parameters_dg_2d
    implicit none
    type (c_ptr) :: device
    integer::icell,jcell,ni,nj,inti,intj, im, ip, ib, it

    call d2h(device,currentsol,nvar*nx*ny*m*m)

    do icell = 1,nx
      do jcell = 1,ny

        call give_bc(icell,jcell,im,ip,ib,it)

        do ni = 1, m
          do nj = 1,m
            do inti = 1, m
              do intj = 1, m

              end do
            end do
          end do
        end do

      end do
    end do

    call h2d(currentsol,device,nvar*nx*ny*m*m)
end subroutine

subroutine give_bc(icell,jcell,im,ip,ib,it)
  use parameters_dg_2d
  implicit none
  integer::icell,jcell,im,ip,ib,it

  im = icell-1
  ip = icell+1
  ib = jcell-1
  it = jcell+1

  if (bc == 1) then
    if (icell == 1) then
        im = nx
    end if

    if (icell == nx) then
        ip = 1
    end if

    if (jcell == ny) then
        it = 1
    end if

    if (jcell == 1) then
        ib = ny
    end if
  end if

end subroutine

subroutine measure_divergence_wasilij(total_divB)
  use parameters_dg_2d
  implicit none

  integer::icell,jcell, inti, intj, ni,nj
  integer::im,ip,it,ib
  real(kind=8)::total_divB, divergence_accum
  real(kind=8)::div_vol, div_surf, dx, dy, divx, divy, legendre_prime, legendre
  real(kind=8)::plus, minus, val
  real(kind=8)::basis_deriv, basis
  real(kind=8),dimension(4)::fc

  div_surf = 0.0
  div_vol = 0.0
  divx = 0.0
  divy = 0.0

  dx = 1.0/nx
  dy = 1.0/ny
  plus = 1.0
  minus = -1.0
  total_divB = 0
  fc = 0
  divergence_accum = 0
  ! volume part
  do icell = 1,nx
    do jcell = 1,ny

      im = icell-1
      ip = icell+1
      ib = jcell-1
      it = jcell+1

      if (icell == 1) then
          im = nx
      end if

      if (icell == nx) then
          ip = 1
      end if

      if (jcell == ny) then
          it = 1
      end if

      if (jcell == 1) then
          ib = ny
      end if

      divergence_accum = divergence_accum +  abs( &
               (currentsol(ip,it,1,1,1) - currentsol(icell,it,1,1,1) + currentsol(ip,jcell,1,1,1)&
                - currentsol(icell,jcell,1,1,1))  &
               +(  currentsol(ip,it,1,1,2) - currentsol(ip,jcell,1,1,2) &
                  + currentsol(icell,it,1,1,2) - currentsol(icell,jcell,1,1,2)) &
              -1/sqrt(dble(3))*(currentsol(ip,it,2,1,2) &
               - currentsol(ip,jcell,2,1,2) - currentsol(icell,it,2,1,2) + currentsol(icell,jcell,2,1,2)) &
              -1/sqrt(dble(3))*(currentsol(ip,it,1,2,1) &
               - currentsol(ip,jcell,1,2,1) - currentsol(icell,it,1,2,1) +currentsol(icell,jcell,1,2,1)))
    end do
  end do

  total_divB = divergence_accum

  print*,'divergence', total_divB
end subroutine