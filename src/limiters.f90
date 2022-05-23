function minmod(a,b,c)
  implicit none
  real(kind=8)::a,b,c,s,minmod,dlim,slop
  s=sign(1d0,a)
  slop = min(abs(b),abs(c))
  dlim = slop
  !if((y*z)<=0.)dlim=0.
  !minmod = s*min(dlim,abs(x))
  if(sign(1d0,b)==s.AND.sign(1d0,c)==s)then
     minmod=s*min(abs(a),abs(b),abs(c))
  else
    minmod=0.0
  endif
  return
end function minmod

function generalized_minmod(a,b,c)
  use parameters_dg_2d
  implicit none
  ! input
  real(kind=8)::a,b,c

  ! output
  real(kind=8)::minmod, dx
  real(kind=8)::generalized_minmod

  ! internal
  real(kind=8)::s

  s = sign(1d0,a)
  dx = boxlen_x/dble(nx)

  if (abs(a) < M*dx**2) then
     generalized_minmod = a
  else
     generalized_minmod = minmod(a,b,c)
  endif

  !write(*,*) minmod
  return
end function generalized_minmod


function minmod2d(u,d_l_x,d_l_y,d_r_x,d_r_y)
  implicit none
  ! input
  real(kind=8)::u,d_l_x,d_l_y,d_r_x,d_r_y

  ! output
  real(kind=8)::minmod2d

  ! internal
  real(kind=8)::s

  s = sign(1d0,u)

  if (sign(1d0,d_l_x) == s .AND. sign(1d0,d_l_y) == s &
      & .and. sign(1d0,d_r_x) == s .and. sign(1d0,d_r_y) == s)  then
     minmod2d = s*min(abs(u),abs(d_l_y),abs(d_l_x),abs(d_r_y),abs(d_r_x))
  else
     minmod2d = 0.0
  end if
  !write(*,*) minmod
  return
end function minmod2d

function limiting(u,ivar,icell,jcell,itop,ibottom,ileft,iright,intnode,jntnode)
   use parameters_dg_2d
   implicit none
   ! input
   real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::u
   integer::ivar,icell,jcell,itop,ibottom,ileft,iright,intnode,jntnode
   !output
   real(kind=8)::limiting
   !internal
   real(kind=8)::d_l_x, d_l_y, d_r_x, d_r_y
   real(kind=8)::coeff_i=1.,coeff_j=1.,coeff_u=1.,central_u=1.
   real(kind=8)::minmod2d
  
   !Write(*,*)icell,jcell,intnode,jntnode
   coeff_j = sqrt(2.0*dble(jntnode-2)+1.0)/sqrt(2.)*sqrt(2.0*dble(intnode-1)+1.0)/sqrt(2.)
   coeff_i = sqrt(2.0*dble(intnode-2)+1.0)/sqrt(2.)*sqrt(2.0*dble(jntnode-1)+1.0)/sqrt(2.)
   
   coeff_j = sqrt(2.0*dble(jntnode-1)+1.0)/sqrt(2.0*dble(jntnode-1)+1.0)
   coeff_i = sqrt(2.0*dble(intnode-1)+1.0)/sqrt(2.0*dble(intnode-1)+1.0)
   
   coeff_u = sqrt(2.0*dble(intnode-1)+1.0)/sqrt(2.)*sqrt(2.0*dble(jntnode-1)+1.0)/sqrt(2.)
   central_u = u(ivar,icell,jcell,intnode,jntnode)
  
   d_r_y = coeff_j*(u(ivar,icell,itop,intnode,jntnode-1)-u(ivar,icell,jcell,intnode,jntnode-1)) 
                                         
   d_l_y = coeff_j*(u(ivar,icell,jcell,intnode,jntnode-1)-u(ivar,icell,ibottom,intnode,jntnode-1))
                    
   d_r_x = coeff_i*(u(ivar,iright,jcell,intnode-1,jntnode)-u(ivar,icell,jcell,intnode-1,jntnode))

   d_l_x = coeff_i*(u(ivar,icell,jcell,intnode-1,jntnode)-u(ivar,ileft,jcell,intnode-1,jntnode))
              
   limiting = minmod2d(central_u, d_r_y, d_l_y, d_r_x, d_l_x)
   !Write(*,*) icell,jcell,intnode,jntnode,central_u,d_r_y,d_l_y,d_r_x,d_l_x,limiting       
   !pause
   return
end function limiting
             
subroutine high_order_limiter(u)
    use parameters_dg_2d
    implicit none

    real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::u, u_new
    real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::w
    real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::nodes, modes
    real(kind=8)::limited,limited2,limiting
    real(kind=8)::d_l_x, d_l_y, d_r_x, d_r_y
    real(kind=8)::coeff,coeff_u,minmod
    integer::i,j,icell,jcell, ivar, itop,ibottom,ileft,iright
    integer::intnode, jntnode
    integer::done
    
    if (m==1) then
      ! no limiting when we only have 1st order approx
      !use_limiter = .false.
      return
    end if

    call compute_primitive(u,w,nx,ny,m)
    ! TODO: implement characteristic variable representation
    ! using Roe average
    !write(*,*) u(1,1,1,:,:)
    u_new = u
    do ivar = 1,nvar
      do icell=1,nx
        do jcell = 1,ny
          done = 0;
          ileft = icell - 1
          iright = icell + 1
          itop = jcell + 1
          ibottom = jcell - 1

          call get_boundary_conditions(ileft,1)
          call get_boundary_conditions(iright,1)
          call get_boundary_conditions(itop,2)
          call get_boundary_conditions(ibottom,2)

          do intnode = m,2,-1
                
            limited = limiting(u,ivar,icell,jcell,itop,ibottom,ileft,iright,intnode,intnode)
            
            if (abs(limited - u(ivar,icell,jcell,intnode,intnode)) < 1e-10) then
                  exit
            end if
            u_new(ivar,icell,jcell,intnode,intnode) = limited
            
            do jntnode = intnode-1,2,-1
              
              limited = limiting(u,ivar,icell,jcell,itop,ibottom,ileft,iright,intnode,jntnode)
              limited2 = limiting(u,ivar,icell,jcell,itop,ibottom,ileft,iright,jntnode,intnode)
              
              if (abs(limited - u(ivar,icell,jcell,intnode,jntnode)) < 1e-10) then
                if (abs(limited2 - u(ivar,icell,jcell,jntnode,intnode)) < 1e-10) then
                  done = 1
                  exit
                end if    
              end if

              u_new(ivar,icell,jcell,intnode,jntnode) = limited
              u_new(ivar,icell,jcell,jntnode,intnode) = limited2
              
            end do
                   
            if (done == 1) then
                  exit
            end if
            coeff = sqrt(2.0*dble(intnode-2)+1.0)/2.
            coeff = sqrt(2.0*dble(intnode-1)-1.0)/sqrt(2.0*dble(intnode-1)+1.0)
            coeff_u = sqrt(2.0*dble(intnode-1)+1.0)/2.
            d_r_y = u(ivar,icell,itop,1,intnode-1) - &
                & u(ivar,icell,jcell,1,intnode-1) 
                          
            d_l_y = u(ivar,icell,jcell,1,intnode-1) - &
                & u(ivar,icell,ibottom,1,intnode-1)
                
            d_r_x = u(ivar,iright,jcell,intnode-1,1) - &
                & u(ivar,icell,jcell,intnode-1,1)

            d_l_x = u(ivar,icell,jcell,intnode-1,1) - &
                & u(ivar,ileft,jcell,intnode-1,1)    
            
            limited = minmod(u(ivar,icell,jcell,1,intnode),coeff*d_r_y,coeff*d_l_y)
            limited2 = minmod(u(ivar,icell,jcell,intnode,1),coeff*d_r_y,coeff*d_l_y)
            !Write(*,*) icell,jcell,intnode,jntnode,d_r_y,d_l_y,d_r_x,d_l_x,limited     
            if (abs(limited2 - u(ivar,icell,jcell,intnode,1)) < 1e-10) then
              if (abs(limited - u(ivar,icell,jcell,1,intnode)) < 1e-10) then
               exit
              end if    
            end if
            u_new(ivar,icell,jcell,intnode,1) = limited
            u_new(ivar,icell,jcell,1,intnode) = limited2
            
          end do
          
       end do
     end do
   end do

  ! Update variables with limited states
  u = u_new
  !write(*,*) u(1,1,1,:,:)
  !pause
  end subroutine high_order_limiter


subroutine compute_limiter(u)
    use parameters_dg_2d
    implicit none
    real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::u
    real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::w
    real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::nodes, modes
    real(kind=8)::limited1,limited2, generalized_minmod
    integer::i,j,icell,jcell, ivar, itop,ibottom,ileft,iright
    integer::intnode, jntnode

    ! look at 1st derivatives, u12, u21
    if (nvar == 1) then
      return
    end if
    if (m==1) then
      ! no limiting when we only have 1st order approx
      !use_limiter = .false.
      return
    end if

    ! mean is given by u11
    if (use_limiter) then
      do ivar = 1,nvar
      do icell=1,nx
        do jcell = 1,ny
          ileft = icell - 1
          iright = icell + 1
          itop = jcell + 1
          ibottom = jcell - 1

          call get_boundary_conditions(ileft,1)
          call get_boundary_conditions(iright,1)
          call get_boundary_conditions(itop,2)
          call get_boundary_conditions(ibottom,2)


          limited1 = generalized_minmod(u(ivar,icell,jcell,1,2), &
          &u(ivar,iright,jcell,1,1)-u(ivar,icell,jcell,1,1),&
          &u(ivar,icell,jcell,1,1)-u(ivar,ileft,jcell,1,1))

          limited2 = generalized_minmod(u(ivar,icell,jcell,2,1),&
          &u(ivar,icell,itop,1,1)-u(ivar,icell,jcell,1,1),&
          &u(ivar,icell,jcell,1,1)-u(ivar,icell,ibottom,1,1))

          if (abs(limited1 - u(ivar,icell,jcell,1,2))<1e-5) then
            u(ivar,icell,jcell,1,2) = limited1
          else
            u(ivar,icell,jcell,1,2:m) = 0.0
          end if

          if (abs(limited2 - u(ivar,icell,jcell,2,1))<1e-5) then
            u(ivar,icell,jcell,2,1) = limited2
          else
            u(ivar,icell,jcell,2:m,1) = 0.0
          end if

      end do
     end do
   end do
  end if

 ! guarantee positivity of density and pressure
 call get_nodes_from_modes(u,nodes)
 call compute_primitive(nodes,w,nx,ny,m)

  do icell=1,nx
    do jcell = 1,ny
      do intnode = 1,m
        do jntnode = 1,m
          if (w(1,icell,jcell,intnode,jntnode)<1e-10) then
            w(1,icell,jcell,intnode,jntnode) = 1e-5
            !write(*,*) 'lim'
          end if

          if (w(4,icell,jcell,intnode,jntnode)<1e-10) then
            w(4,icell,jcell,intnode,jntnode) = 1e-5
          end if

        end do
      end do
    end do
  end do

  call compute_conservative(w,nodes,nx,ny,m)
  call get_modes_from_nodes(nodes, u)
  ! Update variables with limited states

  end subroutine compute_limiter


subroutine limiter_positivity(u)
  use parameters_dg_2d
  implicit none
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::u, u_lim
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::w
  real(kind=8),dimension(1:nvar,1:nx,1:ny,1:m,1:m)::nodes, modes, nodes_cons
  real(kind=8)::u_left,u_right,u_top,u_bottom, u_center, u_deriv
  real(kind=8)::limited1,limited2, generalized_minmod, minmod, legendre, dx
  integer::i,j,icell,jcell, ivar, itop,ibottom,ileft,iright
  integer::intnode, jntnode
  integer::left,right,bottom,top
  
  real(kind=8)::chsi_left=-1,chsi_right=+1
  dx = boxlen_x/dble(nx)
  ! look at 1st derivatives, u12, u21
  
  if (m==1) then
     ! no limiting when we only have 1st order approx
     !use_limiter = .false.
     return
  end if
  
  ! mean is given by u11
  
  ! guarantee positivity of density and pressure
  
  !call get_nodes_from_modes(u,nodes,nx,ny,mx,my)
  call compute_primitive(u,w,nx,ny,m)
  
  u_lim = u
  do ivar = 1,nvar
     !   if ((ivar == 1).or.(ivar==2).or.(ivar==3)) then
     do icell = 1,nx
        do jcell = 1,nx
           !do intnode = 1,1,-1
           left = icell -1
           right = icell+1
           if (icell == 1) then
              left = nx
           else if (icell == nx) then
              right = 1
           end if
           !if (jcell == 1) then
           !   bottom = nx
           !else if (jcell == nx) then
           ! top = 1
           !end if
           
           u_left = u(ivar,left,jcell, 1, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1)+1.0))
           u_right = u(ivar,right,jcell, 1, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1)+1.0))
           u_center = u(ivar,icell,jcell, 1, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1)+1.0))
           u_deriv = u(ivar,icell,jcell, 2, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1+1)+1.0))
           u_lim(ivar,icell,jcell, 2, 1) = minmod(u_deriv,&
                &(u_center-u_left),(u_right-u_center))
           
           if(ABS(u_lim(ivar,icell,jcell, 2, 1)-u_deriv).GT.0.01*ABS(u_deriv)) then
              u_lim(ivar,icell,jcell,2:m,1) = 0.0
              !u_lim(ivar,icell,jcell,mx,mx) = 0.0
           end if
        end do
     end do
     !end if
  end do
  ! end do
  
  do ivar = 1,nvar
     
     !if ((ivar == 1).or.(ivar==2).or.(ivar==3)) then
     do icell = 1,nx
        do jcell=1,nx
           !do intnode = mx-1,1,-1
           left = icell -1
           right = icell+1
           if (icell == 1) then
              left = nx
           else if (icell == nx) then
              right = 1
           end if
           !if (jcell == 1) then
           ! bottom = nx
           !else if (jcell == nx) then
           ! top = 1
           !end if
           
           u_left = u(ivar,jcell,left,1, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1)+1.0))
           u_right = u(ivar,jcell,right,1, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1)+1.0))
           u_center = u(ivar,jcell,icell,1, 1)!*sqrt(dble(2.0))/sqrt((2.0*dble(1)+1.0))
           u_deriv = u(ivar,jcell,icell, 1, 2) !*sqrt(dble(2.0))/sqrt((2.0*dble(1+1)+1.0))
           u_lim(ivar,jcell, icell, 1, 2) =  minmod(u_deriv,&
                &(u_center-u_left),(u_right-u_center))
           if(ABS(u_lim(ivar,jcell,icell, 1,2)-u_deriv).GT.0.01*ABS(u_deriv)) then
              u_lim(ivar,jcell,icell, 1,2:m) = 0.0
              !u_lim(ivar,jcell,icell, mx,mx) = 0.0
           end if
        end do
     end do
     !end if
  end do
  !end do
  
  
  call get_nodes_from_modes(u_lim,nodes)
  call compute_primitive(nodes, nodes_cons, nx, ny, m)
  nodes = nodes_cons
  
  do ivar=1,nvar
     do icell=1,nx
        do jcell=1,ny
           u_left=0.0; u_right=0.0
           !Loop over modes
           !do intnode=1,mx
           do i = 1, m
              do j = 1,m
                 u_left= nodes(ivar,icell,jcell,i,j) !u_left+u_lim(1,icell,jcell,i,j)*legendre(chsi_left,j-1)*legendre(x_quad(intnode),i-1)
                 !u_right= nodes(1,icell,jcell,i,j)  !u_right+u_lim(1,icell,jcell,i,j)*legendre(chsi_right,j-1)*legendre(x_quad(intnode),i-1)
                 if((u_left<1d-6.and.ivar==1).or.(u_left<1d-6.and.ivar==4))then
                    nodes(ivar,icell,jcell,i,j)=1e-5
                    !u_lim(1,icell,jcell,2:mx,2:my)=0.0
                 end if
              end do
           end do
        end do
     end do
  end do
  nodes(4,:,:,:,:) = 1e-4
  
  call compute_conservative(nodes,nodes_cons,nx,ny,m)
  call get_modes_from_nodes(nodes_cons, u_lim)
  ! Update variables with limited states
  u = u_lim
end subroutine limiter_positivity



real(kind=8) function phi_lim(y)
  real(kind=8)::y
  phi_lim = min(1.,y/dble(1.1))
end function phi_lim
