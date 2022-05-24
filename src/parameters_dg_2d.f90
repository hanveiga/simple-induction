module parameters_dg_2d
  ! solver parameter
  !integer,parameter::nx=50
  !integer,parameter::ny=50

  integer::nx,ny,m, k, ninit
  character(LEN=3)::corr_type
  !integer,parameter::m=2
  !integer,parameter::k=(m+3)/2
  integer,parameter::nvar=4
  integer,parameter::riemann=2
  logical,parameter::use_limiter=.false.
  logical,parameter::wb=.false.
  logical,parameter::ldf=.false.
  logical,parameter::corr=.false.
  logical,parameter::make_movie=.true.

  character(LEN=3),parameter::solver='RK4' !or EQL to use the equilibrium solution ^^
  character(LEN=3),parameter::limiter_type='ROS' !or EQL to use the equilibrium solution ^^
  ! Problem set-up
  !integer,parameter::ninit=2
  integer,parameter::bc=1
  integer,parameter::nequilibrium=2
  integer,parameter::source=1

  logical,parameter::ignore=.true. ! ignores the parameter setings specified in this file
  real(kind=8)::tend=1.0
  real(kind=8)::boxlen_x=1.
  real(kind=8)::boxlen_y=1.
  real(kind=8)::gamma=1.4

  real(kind=8)::cfl = 0.4
  real(kind=8)::dtop= 0.01

  ! well balanced parameters
  real(kind=8)::eta=0.0

    real(kind=8),dimension(:),allocatable::sqrts_div

    real(kind=8),dimension(:,:,:,:),allocatable::x,y
    real(kind=8),dimension(:,:),allocatable::xc,yc
    real(kind=8),dimension(:,:,:,:,:),allocatable::w_eq,u_eq


      ! misc commons
    real(kind=8),dimension(:),allocatable::x_quad, w_x_quad
    real(kind=8),dimension(:),allocatable::y_quad, w_y_quad
    real(kind=8),dimension(:),allocatable::x_gll, w_x_gll
    real(kind=8),dimension(:),allocatable::y_gll, w_y_gll
    real(kind=8),dimension(:),allocatable::sqrt_mod
    real(kind=8),dimension(:,:,:,:,:),allocatable::currentsol


  integer,parameter::interval = 10000
  character(len=80)::folder='ind_adv_dis_ldf_o2_t20'
  integer,parameter::dev = 0
end module parameters_dg_2d
