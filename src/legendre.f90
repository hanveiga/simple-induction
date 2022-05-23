function legendre(x,n)
  integer::n
  real(kind=8)::x
  real(kind=8)::legendre
  x=min(max(x,-1.0),1.0)
  select case(n)
  case(0)
     legendre = 1.0
  case(1)
     legendre = x
  case(2)
     legendre = 0.5*(3.0*x**2.0-1.0)
  case(3)
     legendre=0.5*(5.0*x**3.0-3.0*x)
  case(4)
     legendre=0.125*(35.0*x**4-30.0*x**2+3.0)
  case(5)
     legendre=0.125*(63.0*x**5-70.0*x**3+15.0*x)
  case(6)
     legendre=1.0/16.0*(231.0*x**6-315.0*x**4+105.0*x**2-5.0)
  end select
  legendre=sqrt((2.0*dble(n)+1.0))*legendre
  return
end function legendre

function legendre_prime(x,n)
  integer::n
  real(kind=8)::x
  real(kind=8)::legendre_prime
  x=min(max(x,-1.0),1.0)
  select case(n)
  case(0)
     legendre_prime=0.0
  case(1)
     legendre_prime=1.0
  case(2)
     legendre_prime=3.0*x
  case(3)
     legendre_prime=0.5*(15.0*x**2.0-3.0)
  case(4)
     legendre_prime=0.125*(140.0*x**3.0-60.0*x)
  case(5)
     legendre_prime=0.125*(315.0*x**4-210.0*x**2+15.0)
  case(6)
     legendre_prime=1.0/16.0*(1386.0*x**5-1260.0*x**3+210.0*x)
  end select
  legendre_prime=sqrt((2.0*dble(n)+1.0))*legendre_prime
  return
end function legendre_prime

function legendre_prime_prime(x,n)
  integer::n
  real(kind=8)::x
  real(kind=8)::legendre_prime
  x=min(max(x,-1.0),1.0)
  select case(n)
  case(0)
     legendre_prime=0.0
  case(1)
     legendre_prime=0.0
  case(2)
     legendre_prime=3.0
  case(3)
     legendre_prime=0.5*(30.0*x)
  case(4)
     legendre_prime=0.125*(280.0*x**2-60.0)
  case(5)
     legendre_prime=0.125*(4*315.0*x**3-420.0*x**1)
  case(6)
     legendre_prime=1.0/16.0*(1386.0*5*x**4-1260.0*3*x**2+210.0)
  end select
  legendre_prime=sqrt(2.0*dble(n)+1.0)*legendre_prime
  return
end function legendre_prime_prime

subroutine gl_quadrature(x_quad,w_quad,n)
  integer::n
  real(kind=8),dimension(1:n)::x_quad,w_quad

  integer::i,iter
  real(kind=8)::dpi=acos(-1.0d0),xx
  real(kind=8)::legendre,legendre_prime

  !write(*,*)"Computing Gauss-Legendre quadrature points and weights."

  do i=1,n
     xx=(1.0-0.125/dble(n)/dble(n)+0.125/dble(n)/dble(n)/dble(n))* &
          & cos(dpi*(4.0*dble(i)-1.0)/(4.0*dble(n)+2.0))
     do iter=1,500
        xx=xx-legendre(xx,n)/legendre_prime(xx,n)
     end do
     x_quad(i)=-xx
     w_quad(i)=2.0*(2.0*dble(n)+1.0)/(1.0-x_quad(i)**2.0) &
          & /legendre_prime(x_quad(i),n)**2.0
  end do
  do i=n/2+1,n
     x_quad(i)=-x_quad(n-i+1)
     w_quad(i)=w_quad(n-i+1)
  end do

  !do i=1,n
  !   write(*,*)i,x_quad(i),w_quad(i)
  !end do

end subroutine gl_quadrature


subroutine gll_quadrature(x_gll,w_gll,n)
  integer::n
  real(kind=8),dimension(1:n)::x_gll,w_gll
  select case(n)
  case(2)
     x_gll(1) = -1.
     w_gll(1) = 1.
     x_gll(2) = 1.
     w_gll(2) = 1.
  case(3)
     x_gll(1) = -1.
     w_gll(1) = 1./3.
     x_gll(2) = 0.0
     w_gll(2) = 4./3.
     x_gll(3) = 1.
     w_gll(3) = 1./3.
  case(4)
     x_gll(1) = -1.
     w_gll(1) = 1./6.
     x_gll(2) = -1./5.*sqrt(5.)
     w_gll(2) = 5./6.
     x_gll(3) = 1./5.*sqrt(5.)
     w_gll(3) = 5./6.
     x_gll(4) = 1.
     w_gll(4) = 1./6.
  case(5)
     x_gll(1) = -1
     w_gll(1) = 0.1
     x_gll(2) = -sqrt(3.0/7.0)
     w_gll(2) = 4.9/9.0
     x_gll(3) = 0
     w_gll(3) = 3.2/4.5
     x_gll(4) = sqrt(3.0/7.0)
     w_gll(4) = 4.9/9.0
     x_gll(5) = 1.0
     w_gll(5) = 0.1
  case(6)
     x_gll(1) = -1
     w_gll(1) = 1.0/15
     x_gll(2) = -sqrt((1.0/3.0)+(2.0/21.0)*sqrt(7.0))
     w_gll(2) = (14.0-sqrt(7.0))/30.0
     x_gll(3) = -sqrt((1.0/3.0)-(2.0/21.0)*sqrt(7.0))
     w_gll(3) = (14.0+sqrt(7.0))/30.0
     x_gll(4) = sqrt((1.0/3.0)-(2.0/21.0)*sqrt(7.0))
     w_gll(4) = (14.0+sqrt(7.0))/30.0
     x_gll(5) = sqrt((1.0/3.0)+(2.0/21.0)*sqrt(7.0))
     w_gll(5) = (14.0-sqrt(7.0))/30.0
     x_gll(1) = 1
     w_gll(1) = 1.0/15
  end select
end subroutine gll_quadrature

real(kind=8) function lagrange_poly(x_points, y_points, x, n)
  implicit none
  integer::n
  real(kind=8),dimension(1:n)::x_points
  real(kind=8),dimension(1:n)::y_points
  real(kind=8)::x
  real(kind=8)::L
  real(kind=8)::lagrange_basis
  integer::i

  L = 0.0
  do i=1,n
    L = L + dble(y_points(i))*lagrange_basis(x_points, x_points(i), x, n)
  end do
  lagrange_poly = L
end

real(kind=8) function lagrange_basis(points, base_pt, x, n)
  implicit none
  integer::n
  real(kind=8),dimension(1:n)::points
  real(kind=8)::x, base_pt
  real(kind=8)::l, temp
  integer::i

  l = 1.0
  temp = 1.0
  DO i=1,n
    !write(*,*),points(i),base_pt,x
    IF (ABS(points(i)-base_pt)<0.00001) THEN
      temp = temp
      !write(*,*),'skip number'
    ELSE
      l = temp*(x-dble(points(i)))/(dble(base_pt)-dble(points(i)))
      temp = l
      !write(*,*),'l=', l
    END IF
  END DO
  lagrange_basis = l
  end

real(kind=8) function lagrange_prime(x_points, y_points, x, n)
  implicit none
  real(kind=8),dimension(1:n)::x_points
  real(kind=8),dimension(1:n)::y_points
  real(kind=8)::x
  real(kind=8)::L
  real(kind=8)::lagrange_prime_basis
  integer::n,i

  L = 0.0
  do i=1,n
    L = L + y_points(i)*lagrange_prime_basis(x_points, x_points(i), x, n)
  end do
  lagrange_prime = L
end

real(kind=8) function lagrange_prime_basis(points, base_pt, x, n)
  implicit none
  real(kind=8),dimension(1:n)::points
  real(kind=8)::x, base_pt
  real(kind=8)::partial, summed
  integer::n,i,j

  summed = 0.0
  partial = 1.0

  do i = 1,n
    partial = 1.0
    IF (ABS(points(i)-base_pt) <0.00001) THEN
      partial = partial
      cycle
    ELSE
      partial = dble(1)/(base_pt - points(i))
    END IF
    do j = 1,n
      IF (ABS(points(j)-base_pt) < 0.00001) THEN
      ELSE IF (ABS(points(i)-points(j)) < 0.00001) THEN
      ELSE
        partial = partial*(x-points(j))/(base_pt-points(j))
      END IF
      end do
    summed = summed + partial
    end do

  lagrange_prime_basis = summed
end function


real(kind=8) function basis( x,  y, mx, my,  k, var)
  integer::mx,my,k,var
  real(kind=8)::x,y, div_b_basis

  x=min(max(x,-1.0),1.0)
  y=min(max(y,-1.0),1.0)

  select case(var)
    case (0)
      basis = div_b_basis(x,y,mx,var)
    case (1)
      basis = div_b_basis(x,y,mx,var)
  end select
end function

real(kind=8) function basis_deriv( x,  y, mx, my,  k, deriv, var)
  implicit none
  integer::mx,my,k,var,deriv, dim
  real(kind=8)::x,y,ldf_div_basis_prime,legendre

  x=min(max(x,-1.0),1.0)
  y=min(max(y,-1.0),1.0)
  !basis_deriv = 2.0 ! pre assign just for debugging
  select case(var)
    case (0)
      !basis_deriv = legendre(x,y,mx)
      dim = 0
      basis_deriv = ldf_div_basis_prime(x,y,mx,deriv,dim)
    case (1)
      dim = 1
      basis_deriv = ldf_div_basis_prime(x,y,mx,deriv,dim)
    end select
    !write(*,*) 'mx, deriv, var, val', mx, deriv, var, basis_deriv
end function


real(kind=8) function div_b_basis( x,  y,  n,  dim)
  integer::n,dim
  real(kind=8)::x,y

  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);

  if (dim==0) then
    select case(n)
      case (0)
         div_b_basis = 1.0
      case(1)
         div_b_basis = 0.0
      case (2)
         div_b_basis = sqrt(3.)/sqrt(2.)*x
      case (3)
         div_b_basis = sqrt(3.)*y
      case (4)
         div_b_basis = 0.0
       case (5)
          div_b_basis = 1./2.*sqrt(30.)*x*y
       case (6)
          div_b_basis = 0.25*sqrt(30.)*(x*x-1.0/3.0)
      case (7)
         div_b_basis = 0.0!sqrt(3.)/4.*(y*(3.0*sqrt(5.)*x*x-sqrt(5.)-2.0))
      case (8)
         div_b_basis = 0.0!sqrt(7.0)*(3*y*y-1.0)/28.
      case (9)
          div_b_basis = 0.0
      case (10)
          div_b_basis = 0.0!x*sqrt(21.0)*(15*y*y-4.0)/166.0
      case (11)
          div_b_basis = 0.0!x*(95.0*x*x*y*y - 20.0*x*x - 60.0*y*y + 13.0)/120.0
      case (12)
          div_b_basis = 0.0!5.0*x*sqrt(21.0)*(83.0*x*x+21.0*y*y-72.0)/166.
      case (13)
          div_b_basis = 0.0 !9.0*sqrt(35.)*x*y*(3*x*x-2.0)/(20.0*(21.0*sqrt(5.0)-130)**2)
      case (14)
          div_b_basis = 0.0 !sqrt(5.0)*(5*sqrt(7.0)*y*y - sqrt(3.0))*(3*x*x-1.0)/(80*sqrt(21.0)-568)
      case (15)
            div_b_basis = 0.0
      case (16)
            div_b_basis = 0.0
      case (17)
            div_b_basis = 0.0
      case (18)
            div_b_basis = 0.
      case (19)
            div_b_basis = 0.0
      case (20)
            div_b_basis = 0.0
      case (21)
            div_b_basis = 0.0
      case (22)
            div_b_basis = 0.0
      case (23)
            div_b_basis = 0.0

    end select
  else if (dim ==1) then
    select case(n)
      case (0)
         div_b_basis = 0.0
      case (1)
         div_b_basis = -1.0
      case (2)
         div_b_basis = -sqrt(3.)/sqrt(2.)*y
      case (3)
         div_b_basis= 0.0
      case (4)
         div_b_basis= -sqrt(3.)*x
       case (5)
          div_b_basis= -0.25*sqrt(30.)*(y*y-1.0/3.0)
       case (6)
          div_b_basis= -1./2.*sqrt(30.)*x*y
      case (7)
         div_b_basis= 0.0 !-sqrt(3.)/4.*(x*(3.0*sqrt(5.)*y*y-sqrt(5.)-2.0))
      case (8)
         div_b_basis = 0.0
     case (9)
         div_b_basis = 0.0!-sqrt(7.0)*(3*x*x-1.0)/28.0
     case (10)
         div_b_basis =  0.0 !y*sqrt(21.0)*(4*y-5.0*y**2)/166.0
     case (11)
         div_b_basis = 0.0 !-y*(95.0*x*x*y*y-60.0*x*x+20.0*y*y+13.0)/120.
     case (12)
         div_b_basis = 0.0 !-5*y*sqrt(21.0)*(249.0*x*x+7.0*y*y-72.0)/166.
     case (13)
         div_b_basis = 0.0!3*sqrt(7.0)*(27*sqrt(5.0)*y*y*x*x-6*sqrt(5.0)*y*y-9*x*x*sqrt(5.0)+3)/(40*(-130+21*sqrt(5.0))**2)
     case (14)
         div_b_basis = 0.0!-sqrt(5.0)*x*y*(5*sqrt(7.0)*y*y-3*sqrt(3.0))/(40*sqrt(21.0)-284)
     case (15)
       div_b_basis = 0.0
     case (16)
       div_b_basis = 0.0
     case (17)
       div_b_basis = 0.0
     case (18)
       div_b_basis = 0.
     case (19)
       div_b_basis = 0.0
     case (20)
       div_b_basis = 0.0
     case (21)
       div_b_basis = 0.0
     case (22)
       div_b_basis = 0.0
     case (23)
       div_b_basis = 0.0
   end select
   end if
end function

real(kind=8) function ldf_div_basis_prime( x,  y,  n,  deriv,  dim)
  implicit none
  integer::n,dim, deriv
  real(kind=8)::x,y,div_b_basis

  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);
  !div_b_basis = 1.0
  select case(deriv)
    case (0) ! derivative in x
        if (dim==0) then
          select case (n)
            case (0)
               div_b_basis = 0.0
            case (1)
               div_b_basis = 0.0
            case (2)
               div_b_basis = sqrt(3.)/sqrt(2.)
            case (3)
               div_b_basis = 0.0
            case (4)
               div_b_basis = 0.0
            case (5)
               div_b_basis = 6.0*sqrt(5.0)/sqrt(29.)*y
            case (6)
               div_b_basis = 1.0*sqrt(5.)/sqrt(29.)*(6.0*x)
            case (7)
               div_b_basis = sqrt(3.)/4.*(y*(6.0*sqrt(5.)*x-sqrt(5.)))
            case (8)
               div_b_basis = 0.0
          end select
        end if
    case (1) ! deriv in y
       if (dim ==1) then
          select case (n)
            case (0)
               div_b_basis = 0.0
            case (1)
               div_b_basis = 0.0
            case (2)
               div_b_basis = -sqrt(3.)/sqrt(2.)
            case (3)
               div_b_basis= 0.0
            case (4)
               div_b_basis= 0.0
            case (5)
               div_b_basis= -1.0*sqrt(5.0)/sqrt(29.)*(6.0*y) !-sqrt(5.)/sqrt(1126.)*0.5*(6.0*y)
            case (6)
               div_b_basis= -6.0*sqrt(5.0)/sqrt(29.)*x! -sqrt(5.)/sqrt(1126.)*45.*x
            case (7)
               div_b_basis= 0.0 !//-sqrt(3.)/4.*(x*(6.0*sqrt(5.)*y))
            case (8)
               div_b_basis = 0.0
          end select
        end if
      end select
      !write(*,*) 'reached the end'
      !write(*,*) div_b_basis
      ldf_div_basis_prime = div_b_basis
end function


real(kind=8) function div_b_basis_tri( x,  y,  n,  dim)
  integer::n,dim
  real(kind=8)::x,y

  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);

  if (dim==0) then
    select case(n)
      ! order 2
      case (0)
         div_b_basis = 1.0
      case(1)
         div_b_basis = 0.0
      case (2)
         div_b_basis = sqrt(6.)/2.*x
      case (3)
         div_b_basis = sqrt(3.)*y
      case (4)
         div_b_basis = 0.0
         ! order 3
       case (5)
          div_b_basis = sqrt(30.)*(24.*x*x-8.0)/96.
       case (6)
          div_b_basis = -sqrt(30.)*x*y/2.
      case (7)
         div_b_basis = sqrt(5.0)*(24.*y*y-8.0)/16.0
      case (8)
         div_b_basis = 0.0!sqrt(7.0)*(3*y*y-1.0)/28.
         ! order 4
      case (9)
          div_b_basis = sqrt(42.0)*sqrt(83.0)*(5.0*x*x*x-4.0*x)/166.0
      case (10)
          div_b_basis = sqrt(165585.)*(-56.0*x*x*x/83.0-2.0*x*(12.0*y*y-1.0)+410.0*x/83.0)/1824.0
      case (11)
          div_b_basis = sqrt(30.0)*(3.0*x*x*y-1.0*y)/4.0
      case (12)
          div_b_basis = sqrt(7.0)*(5.0*y*y*y-3.0*y)/2.0
      case (13)
          div_b_basis = 0.0
    end select
  else if (dim ==1) then
    select case(n)
      case (0)
         div_b_basis = 0.0
      case (1)
         div_b_basis = 1.0
      case (2)
         div_b_basis = -sqrt(6.)/2.*y
      case (3)
         div_b_basis= 0.0
      case (4)
         div_b_basis= sqrt(3.)*x
         ! order 3
       case (5)
          div_b_basis= -sqrt(30.)*x*y/2.0
       case (6)
          div_b_basis= sqrt(30.)*(24.0*y*y-8.0)/96.0
      case (7)
         div_b_basis= 0.0
      case (8)
         div_b_basis = sqrt(5.0)*(24.0*x*x-8.0)/16.0
         ! order 4
     case (9)
         div_b_basis = sqrt(42.0)*sqrt(83.0)*(-15.*y*x*x+4.0*y)/166.0
     case (10)
         div_b_basis = sqrt(165585.)*(8.*y*y*y + 14.*y*(12.*x*x-1.0)/83.0 - 562.*y/83.)/1824.0 !y*sqrt(21.0)*(4*y-5.0*y**2)/166.0
     case (11)
         div_b_basis = sqrt(30.)*(-3.0*y*y*x + 1.0*x)/4.0 !-y*(95.0*x*x*y*y-60.0*x*x+20.0*y*y+13.0)/120.
     case (12)
         div_b_basis = 0.0 !-5*y*sqrt(21.0)*(249.0*x*x+7.0*y*y-72.0)/166.
     case (13)
         div_b_basis = sqrt(7.0)*(5.0*x*x*x-3.0*x)/2.0
   end select
   end if
   div_b_basis_tri = div_b_basis
end function div_b_basis_tri



real(kind=8) function ldf_div_basis_tri_prime( x,  y,  n,  deriv,  dim)
  implicit none
  integer::n,dim, deriv
  real(kind=8)::x,y,div_b_basis

  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);
  !div_b_basis = 1.0
  select case(deriv)
    case (0) ! derivative in x
        if (dim==0) then
          select case (n)
          case (0)
             div_b_basis = 0.0
          case (1)
             div_b_basis = 0.0
          case (2)
             div_b_basis = sqrt(3.)/sqrt(2.)
          case (3)
             div_b_basis = 0.0
          case (4)
             div_b_basis = 0.0
          ! order 3
          case (5)
             div_b_basis = sqrt(30.)*x/2.
          case (6)
             div_b_basis = -sqrt(30.)*y/2.
          case (7)
             div_b_basis = 0.0
          case (8)
             div_b_basis = 0.0
         ! order 4
         case (9)
             div_b_basis = sqrt(42.0)*sqrt(83.0)*(15.0*x*x-4.0)/166.0
         case (10)
             div_b_basis = sqrt(165585.)*(-168.0*x*x/83.0-2.0*(12.0*y*y-1.0)+410.0/83.0)/1824.0
         case (11)
             div_b_basis = sqrt(30.0)*(6.0*x*y)/4.0
         case (12)
             div_b_basis = 0.0
         case (13)
             div_b_basis = 0.0
         end select
        end if
    case (1) ! deriv in y
       if (dim ==1) then
          select case (n)
            case (0)
               div_b_basis = 0.0
            case (1)
               div_b_basis = 0.0
            case (2)
               div_b_basis = -sqrt(3.)/sqrt(2.)
            case (3)
               div_b_basis= 0.0
            case (4)
               div_b_basis= 0.0
           ! order 3
            case (5)
              div_b_basis=  -sqrt(30.)*x/2.0
            case (6)
              div_b_basis =  sqrt(30.)*y/2.0
           case (7)
              div_b_basis= 0.0
           case (8)
              div_b_basis = 0.0
           ! order 4
           case (9)
              div_b_basis = sqrt(42.0)*sqrt(83.0)*(-15.*x*x+4.0)/166.0
           case (10)
              div_b_basis = sqrt(165585.)*(24.*y*y + 14.*(12.*x*x-1.0)/83.0 - 562./83.)/1824.0
           case (11)
              div_b_basis = sqrt(30.)*(-6.0*y*x)/4.0
           case (12)
              div_b_basis = 0.0
           case (13)
              div_b_basis = 0.0
          end select
        end if
      end select
      !write(*,*) 'reached the end'
      !write(*,*) div_b_basis
      ldf_div_basis_tri_prime = div_b_basis
end function ldf_div_basis_tri_prime
