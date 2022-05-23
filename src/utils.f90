subroutine error_norm(u_numerical, t, error_type)
        use parameters_dg_2d
        implicit none
        real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u_analytical, w, u_numerical
        real(kind=8),dimension(1:nvar)::error
        real(kind=8)::t
        integer::error_type

        call initialisation(u_analytical,w)
        if (error_type==0) then
            call lmax(u_analytical,u_numerical, error)
        else if (error_type==1) then
            call l1norm(u_analytical,u_numerical, error)
        else if (error_type==2) then
            call l2norm(u_analytical,u_numerical, error)
        end if

        write(*,*),'Error type:', error_type, ' ', error

end subroutine error_norm

subroutine lmax(u_analytical,u_numerical, lmaxe)
        use parameters_dg_2d
        implicit none
        real(kind=8),dimension(1:nx,1:nx,1:m,1:m,1:nvar)::u_analytical, u_numerical
        real(kind=8),dimension(1:nvar)::lmaxe

        integer::i,j,ii,jj
        print*,u_numerical(2,2,:,:,1)
        print*,u_analytical(2,2,:,:,1)
        do i = 1,nvar
                lmaxe(i) = maxval(u_analytical(:,:,:,:,i) - u_numerical(:,:,:,:,i))
        end do
        !write(*,*) lmaxe
end subroutine lmax

subroutine l1norm(u_analytical,u_numerical, l1error)
        use parameters_dg_2d
        implicit none
        real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::placeholder, u_analytical, u_numerical
        real(kind=8),dimension(1:nvar)::l1error

        integer::i,j,ii,jj


        do i = 1,nvar
               placeholder(:,:,:,:,i) = abs(u_analytical(:,:,:,:,i)- u_numerical(:,:,:,:,i))
               call kahansum(placeholder(:,:,:,:,i),nx,ny,m,m,l1error(i))
        end do



        l1error = l1error/dble(nx*ny*m*m)

end subroutine l1norm

subroutine l2norm(u_analytical,u_numerical, l2error)
        use parameters_dg_2d
        implicit none
        real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::u_analytical, placeholder, u_numerical
        real(kind=8),dimension(1:nvar)::l2error

        integer::i,j,ii,jj

        do i = 1,nvar
               placeholder(:,:,:,:,i) = (u_analytical(:,:,:,:,i)- u_numerical(:,:,:,:,i))**2
               call kahansum(placeholder(:,:,:,:,i),nx,ny,m,m,l2error(i))
        end do

        l2error = sqrt(l2error)/(nx*ny*m*m)

end subroutine l2norm

subroutine kahansum(tosum,size_x,size_y,order_x,order_y,summed)
   use parameters_dg_2d
   integer::size_x,size_y,order_x,order_y
   real(kind=8),dimension(1:size_x,1:size_y,1:order_x,1:order_y)::tosum
   real(kind=8)::summed,cc,minus_y,tt
   integer::i,j,ii,jj
   summed = 0.0
   cc = 0.0
   minus_y = 0.0
   tt = 0.0

   do i = 1,size_x
           do j=1,size_y
              do ii = 1,order_x
                  do jj = 1,order_y
                        minus_y = tosum(i,j,ii,jj)*w_x_quad(ii)*w_y_quad(jj)/2.0-cc
                        tt = summed + minus_y
                        cc = (tt-summed)-minus_y
                        summed = tt
                  end do
              end do
           end do
  end do

end subroutine kahansum
