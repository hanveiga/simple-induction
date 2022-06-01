#include <stdio.h>
#include "cuda.h"
#include "global.h"
#include <assert.h>

#define BLOCK 256
#define RHO0 1E-10
#define P0 1E-10
#define UPWIND // if not upwind, LLF
#define PRC 1E-16 //Percentage threshold on the high-order limiter
#define LOW_ALPHA //Activate diffusive bound for the high-order limiter
//#define PP //Activate Positivity preserving limiter
#define NR 10 //number of Newton-Raphson iterations on the Positivity preserving limiter
//#define CORR

__constant__ double sqrt_mod[5];
__constant__ double sqrts_div[5];
__constant__ double xquad[5];
__constant__ double yquad[5];
__constant__ double wxquad[5];
__constant__ double wyquad[5];
__constant__ double xgll[6];
__constant__ double ygll[6];
__constant__ double wxgll[6];
__constant__ double wygll[6];

__device__  double legendre(double x, int n, int sq){
  double legendre;
  x=min(max(x,-1.0),1.0);
  switch (n) {
  case 0:
    legendre=1.;
    break;
  case 1:
    legendre=x;
    break;
  case 2:
    legendre=0.5*(3.0*x*x-1.0);
    break;
  case 3:
    legendre=(2.5*x*x*x-1.5*x);
    break;
  case 4:
    legendre=0.125*(35.0*x*x*x*x-30.0*x*x+3.0);
    break;
  case 5:
    legendre=0.125*(63.0*pow(x,5)-70.0*pow(x,3)+15.0*x);
    break;
  case 6:
    legendre=1.0/16.0*(231.0*pow(x,6)-315.0*pow(x,4)+105.0*pow(x,2)-5.0);
    break;
  }
  if(sq==1)
    legendre *= sqrt_mod[n];
  return legendre;
}

__device__  double legendre_prime(double x, int n, int sq){
  double legendre_prime;
  x=min(max(x,-1.0),1.0);
  switch (n) {
  case 0:
    legendre_prime=0.0;
    break;
  case 1:
    legendre_prime=1.0;
    break;
  case 2:
    legendre_prime=3.0*x;
    break;
  case 3:
    legendre_prime=0.5*(15.0*x*x-3.0);
    break;
  case 4:
    legendre_prime=0.125*(140.0*x*x*x-60.0*x);
    break;
  case 5:
    legendre_prime=0.125*(315.0*pow(x,4)-210.0*pow(x,2)+15.0);
    break;
  case 6:
    legendre_prime=1.0/16.0*(1386.0*pow(x,5)-1260.0*pow(x,3)+210.0*x);
    break;
  }
  if(sq==1)
    legendre_prime *= sqrt_mod[n];
  return legendre_prime;
}

__device__  double legendre_vector_basis_c(double x, double y, int n, int dim){
  double div_b_basis;
  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);

  if (dim == 0){
    switch (n){
      case 0:
         div_b_basis = 1.0;
         break;
      case 1:
         div_b_basis = 0.0;
         break;
    }
  }
  else if (dim == 1){
    switch (n){
      case 0:
         div_b_basis = 0.0;
         break;
      case 1:
         div_b_basis= 1.0;
         break;
    }
  }
  return div_b_basis;
}


__device__  double legendre_deriv_vector_basis(double x, double y, int n, int deriv, int dim){
  double div_b_basis;
  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);


  switch(deriv){
    case 0: // derivative in x
      if (dim==0){
        switch (n){
          case 0:
             div_b_basis = 0.0;
             break;
          case 1:
             div_b_basis = 1.0*sqrt_mod[1];
             break;
          case 2:
             div_b_basis = 0.0;
             break;
          case 3:
             div_b_basis = y*sqrt_mod[1]*sqrt_mod[1];
             break;
          case 4: case 5: case 6: case 7: case 8:
             div_b_basis = 0.0;
             break;
        }
      }
      else if (dim == 1){
        switch (n){
          case 0: case 1: case 2: case 3:
             div_b_basis = 0.0;
             break;
          case 4:
             div_b_basis= 0.0;
             break;
          case 5:
             div_b_basis= 1.0*sqrt_mod[1];
             break;
          case 6:
             div_b_basis= 0.0;//-sqrt(5.)/sqrt(1126.)*45.*y;
             break;
          case 7:
             div_b_basis= y*sqrt_mod[1]*sqrt_mod[1];//-sqrt(3.)/4.*((3.0*sqrt(5.)*y*y-sqrt(5.)-2.0));
             break;
          case 8:
             div_b_basis = 0.0;
             break;
        }
      }
      break;

    case 1: // deriv in y
    if (dim==0){
      switch (n){
        case 0:
           div_b_basis = 0.0;
           break;
        case 1:
           div_b_basis = 0.0;
           break;
        case 2:
           div_b_basis = 1.0*sqrt_mod[1];
           break;
        case 3:
           div_b_basis = x*sqrt_mod[1]*sqrt_mod[1];
           break;
        case 4: case 5: case 6: case 7: case 8:
           div_b_basis = 0.0;
           break;
      }
    }
    else if (dim == 1){
      switch (n){
        case 0: case 1: case 2: case 3:
           div_b_basis = 0.0;
           break;
        case 4:
           div_b_basis= 0.0;
           break;
        case 5:
           div_b_basis= 0.0;
           break;
        case 6:
           div_b_basis= 1.0*sqrt_mod[1];//-sqrt(5.)/sqrt(1126.)*45.*y;
           break;
        case 7:
           div_b_basis= x*sqrt_mod[1]*sqrt_mod[1];//-sqrt(3.)/4.*((3.0*sqrt(5.)*y*y-sqrt(5.)-2.0));
           break;
        case 8:
           div_b_basis = 0.0;
           break;
      }
    }
    break;
        }
  return div_b_basis;
}


__device__  double ldf_div_basis(double x, double y, int n, int dim){
  double div_b_basis;
  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);

  if (dim==0){
    switch (n){
      case 0:
         div_b_basis = 1.0;
         break;
      case 1:
         div_b_basis = 0.0;
         break;
      case 2:
         div_b_basis = sqrt(3.)/sqrt(2.)*x;
         break;
      case 3:
         div_b_basis = sqrt(3.)*y;
         break;
      case 4:
         div_b_basis =  0.0;
         break;
      // order 3
      case 5:
         div_b_basis = sqrt(30.)*(3.*x*x-1.0)/12.;
         break;
      case 6:
         div_b_basis = -sqrt(30.)*x*y/2.;//1./29.*sqrt(29.)*sqrt(5.)*(3.*x*x-2.);//sqrt(5.)/sqrt(1126.)*0.5*(3.0*x*x-1.0);
         break;
      case 7:
         div_b_basis = sqrt(5.0)*(3.0*y*y-1.0)/2.0;//0.0;//= 0.25*sqrt(6.)*sqrt(5.)*y*(3.0*x*x-1.0);
         break;
      case 8:
         div_b_basis = 0.0;
         break;
      // order 4
      case 9:
          div_b_basis = sqrt(42.0)*sqrt(83.0)*(5.0*x*x*x-4.0*x)/166.0;
          break;
      case 10:
          div_b_basis = sqrt(165585.)*(-56.0*x*x*x/83.0-2.0*x*(12.0*y*y-1.0)+410.0*x/83.0)/1824.0;
          break;
      case 11:
          div_b_basis = sqrt(30.0)*(3.0*x*x*y-1.0*y)/4.0;
          break;
      case 12:
          div_b_basis = sqrt(7.0)*(5.0*y*y*y-3.0*y)/2.0;
          break;
      case 13:
          div_b_basis = 0.0;
          break;
    }
  }
  else if (dim ==1){
    switch (n){
      case 0:
         div_b_basis = 0.0;
         break;
      case 1:
         div_b_basis = 1.0;
         break;
      case 2:
         div_b_basis = -sqrt(3.)/sqrt(2.)*y;
         break;
      case 3:
         div_b_basis=  0.0;
         break;
      case 4:
         div_b_basis=  sqrt(3.)*x;
         break;
      // order 3
      case 5:
         div_b_basis=  -sqrt(30.)*x*y/2.0;//-sqrt(5.)/sqrt(1126.)*0.5*(3.0*y*y-1.0);
         break;
      case 6:
         div_b_basis=  sqrt(30.)*(3.0*y*y-1.0)/12.0;//-sqrt(5.)/sqrt(1126.)*45.*x*y;
         break;
      case 7:
         div_b_basis= 0.0;//= -0.25*sqrt(6.)*sqrt(5.)*x*(3.0*y*y-1.0);
         break;
      case 8:
         div_b_basis = sqrt(5.0)*(3.0*x*x-1.0)/2.0;
         break;
      //order 4
     case 9:
         div_b_basis = sqrt(42.0)*sqrt(83.0)*(-15.*y*x*x+4.0*y)/166.0;
         break;
     case 10:
         div_b_basis = sqrt(165585.)*(8.*y*y*y + 14.*y*(12.*x*x-1.0)/83.0 - 562.*y/83.)/1824.0;
         break;
     case 11:
         div_b_basis = sqrt(30.)*(-3.0*y*y*x + 1.0*x)/4.0;
         break;
     case 12:
         div_b_basis = 0.0 ;
         break;
     case 13:
         div_b_basis = sqrt(7.0)*(5.0*x*x*x-3.0*x)/2.0;
         break;
    }
  }
  return div_b_basis;
}

__device__  double ldf_div_basis_prime(double x, double y, int n, int deriv, int dim){
  double div_b_basis;
  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);


  switch(deriv){
    case 0: // derivative in x
      if (dim==0){
        switch (n){
          case 0:
             div_b_basis = 0.0;
             break;
          case 1:
             div_b_basis = 0.0;
             break;
          case 2:
             div_b_basis = sqrt(3.)/sqrt(2.);
             break;
          case 3:
             div_b_basis = 0.0;
             break;
          case 4:
             div_b_basis = 0.0;
             break;
          // order 3
          case 5:
             div_b_basis = sqrt(30.)*x/2.;//6./29.*sqrt(29.)*sqrt(5.)*y;//sqrt(5.)/sqrt(1126.)*45.*y;
             break;
          case 6:
             div_b_basis = -sqrt(30.)*y/2.;//sqrt(5.)/sqrt(1126.)*0.5*(6.0*x);
             break;
          case 7:
             div_b_basis = 0.0;// 0.25*sqrt(6.)*sqrt(5.)*y*(6.0*x);//sqrt(3.)/4.*(y*(6.0*sqrt(5.)*x-sqrt(5.)));
             break;
          case 8:
             div_b_basis = 0.0;
             break;
         // order 4
         case 9:
             div_b_basis = sqrt(42.0)*sqrt(83.0)*(15.0*x*x-4.0)/166.0;
             break;
         case 10:
             div_b_basis = sqrt(165585.)*(-168.0*x*x/83.0-2.0*(12.0*y*y-1.0)+410.0/83.0)/1824.0;
             break;
         case 11:
             div_b_basis = sqrt(30.0)*(6.0*x*y)/4.0;
             break;
         case 12:
             div_b_basis = 0.0;
             break;
         case 13:
             div_b_basis = 0.0;
             break;
        }
      }
      else if (dim == 1){
        switch (n){
          case 0:
             div_b_basis = 0.0;
             break;
          case 1:
             div_b_basis = 0.0;
             break;
          case 2:
             div_b_basis = 0.0;
             break;
          case 3:
             div_b_basis= 0.0;
             break;
          case 4:
             div_b_basis= sqrt(3.);
             break;
          // order 3
          case 5:
             div_b_basis=  -sqrt(30.)*y/2.0;
             break;
          case 6:
             div_b_basis=  0.0;
             break;
          case 7:
             div_b_basis= 0.0;
             break;
          case 8:
             div_b_basis = sqrt(5.0)*(6.0*x)/2.0;
             break;
          //order 4
         case 9:
             div_b_basis = sqrt(42.0)*sqrt(83.0)*(-30.*y*x)/166.0;
             break;
         case 10:
             div_b_basis = sqrt(165585.0)*(14.*y*(24.0*x)/83.0)/1824.0;
             break;
         case 11:
             div_b_basis = sqrt(30.)*(-3.0*y*y + 1.0)/4.0;
             break;
         case 12:
             div_b_basis = 0.0 ;
             break;
         case 13:
             div_b_basis = sqrt(7.0)*(15.0*x*x-3.0)/2.0;
             break;
        }
      }
      break;


    case 1: // deriv in y
        if (dim==0){
          switch (n){
            case 0:
               div_b_basis = 0.0;
               break;
            case 1:
               div_b_basis = 0.0;
               break;
            case 2:
               div_b_basis = 0.0;
               break;
            case 3:
               div_b_basis = sqrt(3.);
               break;
            case 4:
               div_b_basis = 0.0;
               break;
            // order 3
            case 5:
               div_b_basis = 0.0;
               break;
            case 6:
               div_b_basis = -sqrt(30.)*x/2.;//1./29.*sqrt(29.)*sqrt(5.)*(3.*x*x-2.);//sqrt(5.)/sqrt(1126.)*0.5*(3.0*x*x-1.0);
               break;
            case 7:
               div_b_basis = sqrt(5.0)*(6.*y)/2.0;//0.0;//= 0.25*sqrt(6.)*sqrt(5.)*y*(3.0*x*x-1.0);
               break;
            case 8:
               div_b_basis = 0.0;
               break;
            // order 4
            case 9:
                div_b_basis = 0.0;
                break;
            case 10:
                div_b_basis = sqrt(165585.0)*(-2.0*x*(24.0*y))/1824.0;
                break;
            case 11:
                div_b_basis = sqrt(30.0)*(3.0*x*x-1.0)/4.0;
                break;
            case 12:
                div_b_basis = sqrt(7.0)*(15.0*y*y-3.0)/2.0;
                break;
            case 13:
                div_b_basis = 0.0;
                break;
          }
        }

        else if (dim ==1){
          switch (n){
            case 0:
               div_b_basis = 0.0;
               break;
            case 1:
               div_b_basis = 0.0;
               break;
            case 2:
               div_b_basis = -sqrt(3.)/sqrt(2.);
               break;
            case 3:
               div_b_basis= 0.0;
               break;
            case 4:
               div_b_basis= 0.0;
               break;
           // order 3
           case 5:
              div_b_basis=  -sqrt(30.)*x/2.0;//-sqrt(5.)/sqrt(1126.)*0.5*(3.0*y*y-1.0);
              break;
           case 6:
              div_b_basis=  sqrt(30.)*y/2.0;//-sqrt(5.)/sqrt(1126.)*45.*x*y;
              break;
           case 7:
              div_b_basis= 0.0;//= -0.25*sqrt(6.)*sqrt(5.)*x*(3.0*y*y-1.0);
              break;
           case 8:
              div_b_basis = 0.0;
              break;
           //order 4
          case 9:
              div_b_basis = sqrt(42.0)*sqrt(83.0)*(-15.*x*x+4.0)/166.0;
              break;
          case 10:
              div_b_basis = sqrt(165585.)*(24.*y*y + 14.*(12.*x*x-1.0)/83.0 - 562./83.)/1824.0;
              break;
          case 11:
              div_b_basis = sqrt(30.)*(-6.0*y*x)/4.0;
              break;
          case 12:
              div_b_basis = 0.0 ;
              break;
          case 13:
              div_b_basis = 0.0;
              break;
          }
        }
          break;
        }
  return div_b_basis;
}


__device__  double basis_ldf_t(double x, double y, int mx, int k, int var){
  double basis;
  int my = 0;
  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);
  switch (var) {
  case 2: case 3:
    if ((mx > k-1) || (my > k-1)){
      basis = 0.0;
      break;
    }
    //msx = mx
    //msy =
    basis = legendre(x,mx,1)*legendre(y,my,1);
    break;
  case 0: //
    basis = ldf_div_basis(x,y,mx,0);
    break;
  case 1: //
    basis = ldf_div_basis(x,y,mx,1);
    break;
  }
  return basis;
}


__device__  double basis_ldf_prime_t(double x, double y, int mx, int k, int deriv, int var){
  double basis;
  x=min(max(x,-1.0),1.0);
  y=min(max(y,-1.0),1.0);
  switch (var) {
  case 2: case 3:
    if ((mx > k-1) || (mx > k-1)){
      basis = 0.0;
      break;
    }
    basis = legendre(x,mx,1)*legendre(y,mx,1);
    break;
  case 0: //
    basis = ldf_div_basis_prime(x,y,mx,deriv,0);
    break;
  case 1: //
    basis = ldf_div_basis_prime(x,y,mx,deriv,1);
    break;
  }
  return basis;
}


__device__  double minmod(double x, double y, double z){
  double s;
  s=copysign(1.0,x);
  if(copysign(1.0,y) == s && copysign(1.0,z) == s)
    return (double)s*min(fabs(x),min(fabs(y),fabs(z)));
  else
     return 0.0;
}

__device__ int BC(int index, int size, int bc){
  if (bc == 1){//periodic
    if (index == -1)
      index = size-1;
    else if (index == size)
      index = 0;
  }
  else if (bc == 2 || bc == 3 || bc == 4){//transmissive or reflective
    if (index == -1)
      index++;
    else if (index == size)
      index--;
  }
  return index;
}

__global__ void get_modes_from_nodes(double* nodes, double* modes, int m, int ny, int nx, int nvar){

  int id, ic, jc, im, jm, var;
  int iq, jq, cid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = nvar*d;
  double val=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  var = id/d;
  ic  = id - var*d;
  jm  = ic/c;
  ic -= jm*c;
  im  = ic/b;
  ic -= im*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a + var*d;

  if( id < size ){
    for( iq=0; iq < m; iq++){
       for( jq=0; jq < m; jq++)
	 val += 0.25*nodes[iq*b+jq*c+cid]*legendre(xquad[iq],im,1)*legendre(yquad[jq],jm,1)
	   * wxquad[iq]*wyquad[jq];
    }
    modes[id] = val;
  }
}

__global__ void get_nodes_from_modes(double* modes, double* nodes, int m, int ny, int nx, int nvar){

  int id, ic, jc, iq, jq, var;
  int im, jm, cid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = nvar*d;
  double val=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  var = id/d;
  ic  = id - var*d;
  jq  = ic/c;
  ic -= jq*c;
  iq  = ic/b;
  ic -= iq*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a + var*d;

  if( id < size ){
    for( im=0; im < m; im++){
       for( jm=0; jm < m; jm++)
	 val += modes[im*b+jm*c+cid]
	   *legendre(xquad[iq],im,1)*legendre(yquad[jq],jm,1);
    }
    nodes[id] = val;
   }
}

__global__ void get_nodes_from_modes_ldf(double* modes, double* nodes, int m, int ny, int nx, int nvar){

  int id, ic, jc, iq, jq, var;
  int im, cid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int bsi = m*(m+3)/2;
  int size = d;
  double val1=0.0;
  double val2=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  var = id/d;
  ic  = id - var*d;
  jq  = ic/c;
  ic -= jq*c;
  iq  = ic/b;
  ic -= iq*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a;

  if( id < size ){
    for( im=0; im < bsi; im++){
        	 val1 += modes[im*b+cid]*basis_ldf_t(xquad[iq],yquad[jq],im,m,0);
           val2 += modes[im*b+cid]*basis_ldf_t(xquad[iq],yquad[jq],im,m,1);
       }
    nodes[id] = val1;
    nodes[id+d] = val2;
    nodes[id+2*d] = 0.0;
    nodes[id+3*d] = 0.0;
   }
}

__global__ void get_modes_from_nodes_ldf(double* nodes, double* modes, int m, int ny, int nx, int nvar){

  int id, ic, jc, mid;
  int iq, jq, cid, modek;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int bsi = m*(m+3)/2;
  int size = nx*ny*bsi;
  double val=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  modek = id/b;
  ic  = id - modek*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a;
  mid = modek;

  if( id < size ){
    for( iq=0; iq < m; iq++){
       for( jq=0; jq < m; jq++){
	       val += 0.25*(nodes[iq*b+jq*c+cid]*basis_ldf_t(xquad[iq],yquad[jq],mid,m,0)
	              + nodes[iq*b+jq*c+cid + d]*basis_ldf_t(xquad[iq],yquad[jq],mid,m,1))
    	          * wxquad[iq]*wyquad[jq];
           }
    }
    modes[id] = val;
  }
}

__global__ void compute_primitive(double* u, double* w, double gamma, int usize, int size){
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    w[id] = u[id];
    w[id+usize] = u[id+usize];
    w[id+2*usize] = u[id+2*usize];
    w[id+3*usize] = u[id+3*usize];
  }
}

__global__ void compute_primitive_t(double* u, double* w, double gamma, int m, int usize, int size){
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    w[id] = u[id];
    w[id+usize] = u[id+usize];
  }
}

__global__ void compute_conservative(double* w, double* u, double gamma, int usize, int size){
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    u[id] = w[id];
    u[id+usize]   =  w[id+usize];
    u[id+2*usize]   =  w[id+2*usize];
    u[id+3*usize] = w[id+3*usize];
  }
}

__global__ void cons_to_prim(double* du,double* dw,double gamma,int m,int ny,int nx,int usize,int size){
  int id, cid, ic, jc, mo;
  double rho,vx,vy,vz,drho,dmx,dmy,dmz,dbx,dby,dbz,bx,by,bz,rho_l;
  int a = nx;
  int b = ny*a;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  mo  = id/b;
  ic  = id - mo*b;
  jc  = ic/a;
  ic -= jc*a;
  cid = ic + jc*a;
  if( id < size ){
    id += b;
    drho = dw[id] = du[id];
    rho = dw[cid];
    rho_l = max(rho,rho);
    // central values
    vx = dw[cid+usize];
    vy = dw[cid+usize*2];
    vz = dw[cid+usize*3];

    bx = dw[cid+usize*4];
    by = dw[cid+usize*5];
    bz = dw[cid+usize*6];

    dmx = du[id+usize];
    dmy = du[id+usize*2];
    dmz = du[id+usize*3];

    // velocities
    dw[id+usize]   = (dmx-drho*vx)/rho_l;
    dw[id+usize*2] = (dmy-drho*vy)/rho_l;
    dw[id+usize*3] = (dmz-drho*vz)/rho_l;

    // magnetic field
    dw[id+usize*4] = dbx = du[id+usize*4];
    dw[id+usize*5] = dby = du[id+usize*5];
    dw[id+usize*6] = dbz = du[id+usize*6];

    // pressure
    dw[id+usize*7] = (gamma-1.0)*(du[id+usize*7] + 0.5*drho*(vx*vx+vy*vy+vz*vz)
                                -(vx*dmx+vy*dmy+vz*dmz)
                                -(bx*dbx+by*dby+bz*dbz));

  }
}

__global__ void prim_to_cons(double* dw,double* du,double gamma,int m,int ny,int nx,int usize,int size){
  int id, cid, ic, jc, mo;
  double rho,vx,vy,vz,drho,dvx,dvy,dvz,bx,by,bz,dbx,dby,dbz, rho_l;
  int a = nx;
  int b = ny*a;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  mo = id/b;
  ic  = id - mo*b;
  jc  = ic/a;
  ic -= jc*a;
  cid = ic + jc*a;
  if( id < size ){
    id += b;
    drho = du[id] = dw[id];

    // central values
    rho = dw[cid];
    rho_l = max(rho,rho);
    vx = dw[cid+usize];
    vy = dw[cid+usize*2];
    vz = dw[cid+usize*3];
    bx = dw[cid+usize*4];
    by = dw[cid+usize*5];
    bz = dw[cid+usize*6];

    dvx = dw[id+usize];
    dvy = dw[id+usize*2];
    dvz = dw[id+usize*3];

    dbx = du[id + usize*4] = dw[id + usize*4];
    dby = du[id + usize*5] = dw[id + usize*5];
    dbz = du[id + usize*6] = dw[id + usize*6];

    du[id+usize]   = vx*drho+rho_l*dvx;
    du[id+usize*2] = vy*drho+rho_l*dvy;
    du[id+usize*3] = vz*drho+rho_l*dvz;

    du[id+usize*7] = 0.5*drho*(vx*vx+vy*vy+vz*vz)+rho_l*(vx*dvx+vy*dvy+vz*dvz)
                      + dw[id+usize*7]/(gamma-1.0)
                      + (bx*dbx+by*dby+bz*dbz);
  }
}

__global__ void compute_flux(double* u, double* w, double* flux1, double* flux2, int size){
  int id;
  double psi;
  double b1, b2, b3, v1, v2, v3;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    b1  = w[id];
    b2  = w[id+size];
    b3  = w[id+2*size];
    v1  = 1.0;
    v2  = 1.0;
    v3  = 0.0;
    psi = w[id+size*3];

    flux1[id] = 0.0;//psi;
    flux1[id+size] = (v1*b2 - v2*b1);
    flux1[id+2*size] = 0.0;

    flux2[id] = (v2*b1-v1*b2);
    flux2[id+size] = 0.0;
    flux2[id+2*size] = 0.0;

    flux1[id+size*3] = 0.0;
    flux2[id+size*3] = 0.0;


   }
}

__global__ void compute_flux_b(double* u, double* w, double* flux1, double* flux2, double ch, int m, int size){
  int id;
  double psi;
  double b1, b2, b3, v1, v2, v3;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    b1  = w[id];
    b2  = w[id+size];
    b3  = w[id+2*size];
    v1  = 1.0;
    v2  = 1.0;
    v3  = 0.0;
    psi = u[id+size*3];

    flux1[id+size*3] = ch*ch*b1;
    flux2[id+size*3] = ch*ch*b2;
    flux1[id+size] = 0.0;
    flux2[id] = 0.0;
    flux1[id] = psi;
    flux2[id+size] = psi;

   }
}

__global__ void flux_vol (double* f_vol, double* f_q1, double* f_q2, double invdx, double invdy, int m, int ny, int nx, int nvar){
  int id, ic, jc, iq, jq, va;
  int im, jm, cid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = nvar*d;
  double val1,val2;
  val1=val2=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  va = id/d;
  ic  = id - va*d;
  jm  = ic/c;
  ic -= jm*c;
  im  = ic/b;
  ic -= im*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a + va*d;
  if( id < size ){
    for( iq=0; iq < m; iq++){
      for( jq=0; jq < m; jq++){
	val1 += f_q1[iq*b+jq*c+cid]*
	  legendre_prime(xquad[iq],im,1)*wxquad[iq]*
	  legendre(yquad[jq],jm,1)*wyquad[jq];
	val2 += f_q2[iq*b+jq*c+cid]*
	  legendre_prime(yquad[jq],jm,1)*wyquad[jq]*
	  legendre(xquad[iq],im,1)*wxquad[iq];
      }
    }
    f_vol[id] = val1*invdx+val2*invdy;
  }
}

__global__ void flux_vol_ldf_t (double* f_vol, double* F, double* G, double invdx, double invdy, int m, int ny, int nx, int nvar){
  int id, ic, jc, iq, jq, va;
  int im, jm, cid, mid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int modek;
  int bsi = m*(m+3)/2;
  int size = bsi*nx*ny;
  double f1,f2,g1,g2;
  f1=f2=g1=g2=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  modek = id/(b);
  ic  = id - modek*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a;
  mid = modek;

  if( id < size ){
    for( iq=0; iq < m; iq++){
      for( jq=0; jq < m; jq++){
      	f1 += 0.0;
        g1 += G[iq*b+jq*c+cid]*basis_ldf_prime_t(xquad[iq],yquad[jq],mid,m,1,0)*wxquad[iq]*wyquad[jq];

        f2 += F[iq*b+jq*c+cid+d]*basis_ldf_prime_t(xquad[iq],yquad[jq],mid,m,0,1)*wxquad[iq]*wyquad[jq];
        g2 += 0.0;


      }
    }
    f_vol[id] = f2*invdx + g1*invdy;
  }
}

__global__ void volume_integral (double* f_vol, double* f, double coeff, double invdx, double invdy, int m, int ny, int nx, int nvar){
  int id, ic, jc, iq, jq, va;
  int im, jm, cid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = nvar*d;
  double val1,val2;
  val1=val2=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  va = id/d;
  ic  = id - va*d;
  jm  = ic/c;
  ic -= jm*c;
  im  = ic/b;
  ic -= im*b;
  jc =  ic/a;
  ic -= jc*a;
  cid = ic + jc*a + va*d;
  if( id < size ){
    for( iq=0; iq < m; iq++){
      for( jq=0; jq < m; jq++){
        //val1 += coeff*f[iq*b+jq*c+cid]*
        //  legendre(xquad[iq],im,1)*wxquad[iq]*
        //  legendre(yquad[jq],jm,1)*wyquad[jq];
        val1 += coeff*f[iq*b+jq*c+cid]*
          legendre(xquad[iq],im,1)*wxquad[iq]*
          legendre(yquad[jq],jm,1)*wyquad[jq];
      }
    }
    f_vol[id] = val1;
  }
}

__global__ void compute_min_dt(double* w, double* Dt, double gamma, double cfl, double dx, double dy, int m, int usize, int size){
  int id, jump;
  double dt,dt_min,csx,csy,constant,cs,d2;
  __shared__ double mins[BLOCK];
  id = threadIdx.x;
  if(id < size){
     constant = cfl/double(2*m-1);

    cs = gamma;
    d2 = 0.5*((w[id]*w[id] + w[id+usize]*w[id+usize]+w[id+usize*2]*w[id+usize*2])/1.0 + cs);
    //csx = sqrt(d2 + sqrt(d2*d2-cs*w[id+usize*3]*w[id+usize*3]/w[id]));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
    //csy = sqrt(d2 + sqrt(d2*d2-cs*w[id+usize*4]*w[id+usize*4]/w[id]));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
    csx = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
    csy = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));

    //dt_min = constant/((fabs(w[id+usize])+csx)/dx + (fabs(w[id+usize*2])+csy)/dy);
    //dt_min = constant/((1.0+csx)/dx + (1.0+csy)/dy);
    dt_min = constant/((1.0)/dx + (1.0)/dy);
    for (id = threadIdx.x+blockDim.x; id < size; id += blockDim.x){ //This is implemented considering only one block in the reduction launch.
      cs = gamma;
      d2 = 0.5*((w[id]*w[id] + w[id+usize]*w[id+usize]+w[id+usize*2]*w[id+usize*2])/1.0 + cs);
      //csx = sqrt(d2 + sqrt(d2*d2-cs*w[id+usize*3]*w[id+usize*3]/w[id]));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
      //csy = sqrt(d2 + sqrt(d2*d2-cs*w[id+usize*4]*w[id+usize*4]/w[id]));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
      csx = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
      csy = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
      //dt = constant/((1.0+csx)/dx + (1.0+csy)/dy);
      dt = constant/((1.0)/dx + (1.0)/dy);

      dt_min=min(dt,dt_min);
    }
    mins[threadIdx.x] = dt_min;
  }
  __syncthreads();
  for(jump = blockDim.x/2; jump > 0; jump >>= 1){
    if( threadIdx.x < jump )
      mins[threadIdx.x]=min(mins[threadIdx.x],mins[threadIdx.x+jump]);
    __syncthreads();
  }
  if(threadIdx.x == 0)
    Dt[0] = mins[0];
}

__global__ void compute_bounds(double* mat, double* bounds, int m, int nx, int ny, int nvar, int size,int var){
  int id,jump;
  int cid, jump2;
  double p,p_max,p_min;
  __shared__ double mins[BLOCK],maxs[BLOCK];
  id = threadIdx.x;
  if(id < size){
    cid = id + size * var;
    p_min = mat[cid];
    p_max = mat[cid];
    for (id = threadIdx.x+blockDim.x; id < size; id += blockDim.x){ //This is implemented considering only one block in the reduction launch.
      p = mat[cid];
      p_min=min(p,p_min);
      p_max=max(p,p_max);
    }
    mins[threadIdx.x] = p_min;
    maxs[threadIdx.x] = p_max;
  }
  __syncthreads();
  for(jump = blockDim.x/2; jump > 0; jump >>= 1){
    if( threadIdx.x < jump ){
      mins[threadIdx.x]=min(mins[threadIdx.x],mins[threadIdx.x+jump]);
      maxs[threadIdx.x]=max(maxs[threadIdx.x],maxs[threadIdx.x+jump]);
     }
    __syncthreads();
  }
  if(threadIdx.x == 0){
    bounds[0] = mins[0];
    bounds[1] = maxs[0];}
}

extern "C" void get_bounds_ (double* mat, int size){
 // size = nx*ny*m*m;
  double bound[2];
  for(int var = 0;var<nvar;var++){
        cudaMemset(pivot1,0.0,tsize*sizeof(double));
 	compute_bounds<<<1,BLOCK>>>( mat, pivot1, m, nx, ny, nvar, size, var);
  	cudaMemcpy(&bound,pivot1,2*sizeof(double),cudaMemcpyDeviceToHost);
  	printf("\n var: %i min: %.14g max: %.14g \n",var, bound[0], bound[1]);
	}
}

__global__ void compute_faces(double* ufaces, double* delta_u, int m, int ny, int nx, int nvar){
  int id, va, ic, jc, im, jm, q, cid, lid, rid, bid, tid;
  double shudl[5],shudr[5],shudb[5],shudt[5];
  double chsi_m = -1, chsi_p = 1, du;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = nvar*b;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  va = id/b;
  ic = id - va*b;
  jc = ic/a;
  ic -= jc*a;
  cid = ic + jc*a + va*d;
  d = 4*c;
  lid = ic + jc*a + va*d;
  rid = lid + c;
  bid = rid + c;
  tid = bid + c;
  if( id < size ){
    for (q=0;q<m;q++)
      shudb[q] = shudt[q] = shudl[q] = shudr[q] = 0.0;
    for (im=0;im<m;im++){
      for (jm=0;jm<m;jm++){
	du = delta_u[im*b+jm*c+cid];
	for (q=0;q<m;q++){
	  shudl[q] += du*legendre(chsi_m,im,1)*legendre(yquad[q],jm,1);
	  shudr[q] += du*legendre(chsi_p,im,1)*legendre(yquad[q],jm,1);
	  shudb[q] += du*legendre(chsi_m,jm,1)*legendre(xquad[q],im,1);
	  shudt[q] += du*legendre(chsi_p,jm,1)*legendre(xquad[q],im,1);
	}
      }
    }
    for (q=0;q<m;q++){
      ufaces[lid+q*b]=shudl[q];
      ufaces[rid+q*b]=shudr[q];
      ufaces[bid+q*b]=shudb[q];
      ufaces[tid+q*b]=shudt[q];
    }
  }
}

__global__ void compute_faces_ldf_t(double* ufaces, double* modesu, int m, int ny, int nx, int nvar){
  int id, va, ic, jc, im, jm, q, cid, lid, rid, bid, tid;
  double shudl1[5],shudr1[5],shudb1[5],shudt1[5];
  double shudl2[5],shudr2[5],shudb2[5],shudt2[5];
  double chsi_m = -1, chsi_p = 1, du;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = b;
  int bsi = m*(m+3)/2;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  ic = id;
  jc = ic/a;
  ic -= jc*a;
  cid = ic + jc*a;// + va*d;
  d = 4*c;
  lid = ic + jc*a;// + va*d;
  rid = lid + c;
  bid = rid + c;
  tid = bid + c;
  if( id < size ){
    for (q=0;q<m;q++){
      shudb1[q] = shudt1[q] = shudl1[q] = shudr1[q] = 0.0;
      shudb2[q] = shudt2[q] = shudl2[q] = shudr2[q] = 0.0;}
    for (im=0;im<  bsi;im++){
      //for (jm=0;jm<m;jm++){
	       du = modesu[im*b+cid];
        	for (q=0;q<m;q++){
        	  shudl1[q] += du*basis_ldf_t(chsi_m,yquad[q],im,m,0);//legendre(chsi_m,im,1)*legendre(yquad[q],jm,1);
        	  shudr1[q] += du*basis_ldf_t(chsi_p,yquad[q],im,m,0);//legendre(chsi_p,im,1)*legendre(yquad[q],jm,1);
        	  shudb1[q] += du*basis_ldf_t(xquad[q],chsi_m,im,m,0);//legendre(chsi_m,jm,1)*legendre(xquad[q],im,1);
        	  shudt1[q] += du*basis_ldf_t(xquad[q],chsi_p,im,m,0);//legendre(chsi_p,jm,1)*legendre(xquad[q],im,1);

            shudl2[q] += du*basis_ldf_t(chsi_m,yquad[q],im,m,1);//legendre(chsi_m,im,1)*legendre(yquad[q],jm,1);
        	  shudr2[q] += du*basis_ldf_t(chsi_p,yquad[q],im,m,1);//legendre(chsi_p,im,1)*legendre(yquad[q],jm,1);
        	  shudb2[q] += du*basis_ldf_t(xquad[q],chsi_m,im,m,1);//legendre(chsi_m,jm,1)*legendre(xquad[q],im,1);
        	  shudt2[q] += du*basis_ldf_t(xquad[q],chsi_p,im,m,1);//legendre(chsi_p,jm,1)*legendre(xquad[q],im,1);
	             }
          //}
      }
    for (q=0;q<m;q++){
      ufaces[lid+q*b]=shudl1[q];
      ufaces[rid+q*b]=shudr1[q];
      ufaces[bid+q*b]=shudb1[q];
      ufaces[tid+q*b]=shudt1[q];

      ufaces[lid+q*b+d]=shudl2[q];
      ufaces[rid+q*b+d]=shudr2[q];
      ufaces[bid+q*b+d]=shudb2[q];
      ufaces[tid+q*b+d]=shudt2[q];

        ufaces[lid+q*b+d*2]=0.0;
        ufaces[rid+q*b+d*2]=0.0;
        ufaces[bid+q*b+d*2]=0.0;
        ufaces[tid+q*b+d*2]=0.0;

        ufaces[lid+q*b+d*3]=0.0;
        ufaces[rid+q*b+d*3]=0.0;
        ufaces[bid+q*b+d*3]=0.0;
        ufaces[tid+q*b+d*3]=0.0;



    }
  }
}

__forceinline__ __device__ double exact_sol(double x, double y, int va, int neql){
	double sol, r;
	switch (neql){
	  case 2:{
	        if (va == 0){
		  sol = exp(-(x+y));}
        	else if (va == 1){
	  	  sol = 0;}
		else if (va == 2){
	  	  sol = 0;}
		else if (va ==3){
	  	  sol = exp(-(x+y));
		}
    	  	break;}
	  case 17:{
		double rho_0, r0, n, grav, Ms, eps, h, vk, cs, vt, omega;
    		rho_0 = 1;
    		r0 = 0.275*10.;
    		n = 10;
    		grav = 1;
    		Ms = 1;
    		eps = 1E-1;
    		h = 0.03;
            	r = sqrt(pow(x,2.) + pow(y,2.));
            	omega = sqrt(1./pow((r*r + eps*eps),3./2.));
            	vk = sqrt(r/pow((r*r+eps*eps),3./2.));
            	cs = h*vk;
            	vt = sqrt(-(-2*h*h*r/pow(eps*eps+r*r,3./2.)+3*h*h/(eps*eps+r*r))*r+(r*r)/pow((eps*eps+r*r),3./2.));
 	        if (va == 0){sol = rho_0/(1.+(r/r0)*(r/r0));}
        	else if (va == 1){sol = -vt/r*y;}
		else if (va == 2){sol = vt/r*x;}
		else if (va ==3){sol = cs*cs*rho_0/(1.+(r/r0)*(r/r0));}
		break;}
	  case 19:{
	        if (va == 0){
		  sol = exp(-(x));}
        	else if (va == 1){
	  	  sol = 0;}
		else if (va == 2){
	  	  sol = 0;}
		else if (va ==3){
	  	  sol = exp(-(x));
		}
    	  	break;}
	  default:{
	    	break;}
  	}

	return sol;
}

__global__ void compute_GxGLL(double* nodesX, double* nodesY, double* modes, int k, int m, int ny, int nx, int nvar){
  int id, va, ic, jc, gl, gll, iq, jq, cid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = k*c;
  int size = d*nvar;
  double valx,valy,u;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  va = id/d;
  ic  = id - va*d;
  gll = ic/c;
  ic -= gll*c;
  gl  = ic/b;
  ic -= gl*b;
  jc  = ic/a;
  ic -= jc*a;

  c = m*b;
  d = m*c;
  cid = ic + jc*a + va*d;

  if( id < size ){
    valx=valy=0.0;
    for(iq=0;iq<m;iq++){
      for(jq=0;jq<m;jq++){
	u=modes[iq*b+jq*c+cid];
	valx += u*legendre(xgll[gll],iq,1)*legendre(yquad[gl],jq,1);
	valy += u*legendre(ygll[gll],jq,1)*legendre(xquad[gl],iq,1);
      }
    }
    nodesX[id]=valx;
    nodesY[id]=valy;
  }
}

__global__ void compute_LLF(double* u, double* w, double* f, double* FG,
			   double gamma, int m, int ny, int nx, int nvar, int dim, int bc, int size){
  int id, cid, var, cell, face, quad, mc, pc, pid, mid, fsize;
  double speed_m, speed_p, cmax;
  double bnormp, bnormm, c2p, c2m, d2p, d2m;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  int a = nx+1-dim;
  int b = (ny+dim)*a;
  int c = m*b;
  int d;
  quad = id/b;
  if(dim == 0){
    face = id-quad*b;
    cell = face/a;
    face -= cell*a;
    cid = cell*nx + quad*nx*ny;
    fsize = nx;
    a = 1;
  }
  else if(dim == 1){
    cell = id-quad*b;
    face = cell/a;
    cell -= face*a;
    cid = cell + quad*nx*ny;
    fsize = ny;
    a = nx;
  }
  b=m*nx*ny*(2*dim);
  c=m*nx*ny*(2*dim+1);
  d=4*m*nx*ny;
  if(id < size){
    mc = BC(face-1,fsize,bc);
    pc = BC(face,fsize,bc);
    mid = cid+mc*a+c;
    pid = cid+pc*a+b;

    bnormp = w[pid+d*5]*w[pid+d*5] + w[pid+d*4]*w[pid+d*4]+w[pid+d*6]*w[pid+d*6];
    bnormm = w[mid+d*5]*w[mid+d*5] + w[mid+d*4]*w[mid+d*4]+w[mid+d*6]*w[mid+d*6];
    c2p = gamma*w[pid+d*7]/w[pid];
    c2m = gamma*w[mid+d*7]/w[mid];
    d2p = 0.5*(bnormp/w[pid] + c2p);
    d2m = 0.5*(bnormm/w[mid] + c2m);

    speed_p = fabs(w[pid+d*(dim+1)])+ sqrt(d2p + sqrt(d2p*d2p-c2p*w[pid+d*(4+dim)]*w[pid+d*(4+dim)]/w[pid])); //sqrt(gamma*max(w[pid+d*3],P0)/max(w[pid],RHO0)));
    speed_m = fabs(w[mid+d*(dim+1)])+ sqrt(d2m + sqrt(d2m*d2m-c2m*w[mid+d*(4+dim)]*w[mid+d*(4+dim)]/w[mid]));//sqrt(gamma*max(w[mid+d*3],P0)/max(w[mid],RHO0)));

    cmax=max(speed_m,speed_p);
    for(var = 0; var < nvar; var++){
      FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var])-0.5*cmax*(u[pid+d*var]-u[mid+d*var]);
    }
  }
}

__global__ void compute_LLF_fixed_speed(double* u, double* w, double* f, double* FG,
			   double gamma, int m, int ny, int nx, int nvar, int dim, int bc, int size){
  int id, cid, var, cell, face, quad, mc, pc, pid, mid, fsize;
  double speed_m, speed_p, cmax;
  double bnormp, bnormm, c2p, c2m, d2p, d2m;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  int a = nx+1-dim;
  int b = (ny+dim)*a;
  int c = m*b;
  int d;
  quad = id/b;
  if(dim == 0){
    face = id-quad*b;
    cell = face/a;
    face -= cell*a;
    cid = cell*nx + quad*nx*ny;
    fsize = nx;
    a = 1;
  }
  else if(dim == 1){
    cell = id-quad*b;
    face = cell/a;
    cell -= face*a;
    cid = cell + quad*nx*ny;
    fsize = ny;
    a = nx;
  }
  b=m*nx*ny*(2*dim);
  c=m*nx*ny*(2*dim+1);
  d=4*m*nx*ny;
  if(id < size){
    mc = BC(face-1,fsize,bc);
    pc = BC(face,fsize,bc);
    mid = cid+mc*a+c;
    pid = cid+pc*a+b;

    bnormp = w[pid]*w[pid] + w[pid+d]*w[pid+d];//+w[pid+d*2]*w[pid+d*2];
    bnormm = w[mid]*w[mid] + w[mid+d]*w[mid+d];//+w[mid+d*2]*w[mid+d*2];
    c2p = gamma;
    c2m = gamma;
    d2p = 0.5*(bnormp + c2p);
    d2m = 0.5*(bnormm + c2m);

    speed_p = fabs(1.0);// + sqrt(d2p + sqrt(d2p*d2p-c2p)); //sqrt(gamma*max(w[pid+d*3],P0)/max(w[pid],RHO0)));
    speed_m = fabs(1.0);// + sqrt(d2m + sqrt(d2m*d2m-c2m));//sqrt(gamma*max(w[mid+d*3],P0)/max(w[mid],RHO0)));

    cmax=max(speed_m,speed_p);

    for(var = 0; var < nvar; var++){

	  FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var])-0.5*cmax*(u[pid+d*var]-u[mid+d*var]);
    //FG[id+size*var]=f[mid+d*var];

    }
  }
}

__global__ void compute_true_upwind(double* u, double* w, double* f, double* FG,
			   double gamma, int m, int ny, int nx, int nvar, int dim, int bc, int size){
  // Roe flux 
  int id, cid, var, cell, face, quad, mc, pc, pid, mid, fsize;
  double speed_m, speed_p, cmax;
  double bnormp, bnormm, c2p, c2m, d2p, d2m;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  int a = nx+1-dim;
  int b = (ny+dim)*a;
  int c = m*b;
  int d;
  quad = id/b;
  if(dim == 0){
    face = id-quad*b;
    cell = face/a;
    face -= cell*a;
    cid = cell*nx + quad*nx*ny;
    fsize = nx;
    a = 1;
  }
  else if(dim == 1){
    cell = id-quad*b;
    face = cell/a;
    cell -= face*a;
    cid = cell + quad*nx*ny;
    fsize = ny;
    a = nx;
  }
  b=m*nx*ny*(2*dim);
  c=m*nx*ny*(2*dim+1);
  d=4*m*nx*ny;
  if(id < size){
    mc = BC(face-1,fsize,bc);
    pc = BC(face,fsize,bc);
    mid = cid+mc*a+c;
    pid = cid+pc*a+b;

    bnormp = w[pid]*w[pid] + w[pid+d]*w[pid+d];//+w[pid+d*2]*w[pid+d*2];
    bnormm = w[mid]*w[mid] + w[mid+d]*w[mid+d];//+w[mid+d*2]*w[mid+d*2];
    c2p = gamma;
    c2m = gamma;
    d2p = 0.5*(bnormp + c2p);
    d2m = 0.5*(bnormm + c2m);

    speed_p = fabs(1.0);// + sqrt(d2p + sqrt(d2p*d2p-c2p)); //sqrt(gamma*max(w[pid+d*3],P0)/max(w[pid],RHO0)));
    speed_m = fabs(1.0);// + sqrt(d2m + sqrt(d2m*d2m-c2m));//sqrt(gamma*max(w[mid+d*3],P0)/max(w[mid],RHO0)));

    cmax=max(speed_m,speed_p);

    for(var = 0; var < nvar; var++){

      // if in Bx --
      if ((var == 0)&&(dim==0)){
        //FG[id+size*var]=f[mid+d*var];
        FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var]); //-0.5*cmax*(u[pid+d*var]-u[mid+d*var]);
      }
      if ((var == 0)&&(dim==1)){
        //FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var]);
        FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var]) - 0.5*(u[pid+d*var]-u[mid+d*var]) + 0.5*(u[pid+d*1]-u[mid+d*1]);
      }
      // if in By
      if ((var == 1)&&(dim==0)){
        FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var]) + 0.5*(u[pid+d*0]-u[mid+d*0]) - 0.5*(u[pid+d*var]-u[mid+d*var]);
        //FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var]);
      }
      if ((var == 1)&&(dim==1)){
        //FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var])-0.5*cmax*(u[pid+d*var]-u[mid+d*var]);
        FG[id+size*var]=0.5*(f[pid+d*var]+f[mid+d*var]);
      }

    }
  }
}

__global__ void compute_average(double* u, double* w, double* f, double* FG,
			   double gamma, int m, int ny, int nx, int nvar, int dim, int bc, int size){
  int id, cid, var, cell, face, quad, mc, pc, pid, mid, fsize;
  double speed_m, speed_p, cmax;
  double bnormp, bnormm, c2p, c2m, d2p, d2m;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  int a = nx+1-dim;
  int b = (ny+dim)*a;
  int c = m*b;
  int d;
  quad = id/b;
  if(dim == 0){
    face = id-quad*b;
    cell = face/a;
    face -= cell*a;
    cid = cell*nx + quad*nx*ny;
    fsize = nx;
    a = 1;
  }
  else if(dim == 1){
    cell = id-quad*b;
    face = cell/a;
    cell -= face*a;
    cid = cell + quad*nx*ny;
    fsize = ny;
    a = nx;
  }
  b=m*nx*ny*(2*dim);
  c=m*nx*ny*(2*dim+1);
  d=4*m*nx*ny;
  if(id < size){
    mc = BC(face-1,fsize,bc);
    pc = BC(face,fsize,bc);
    mid = cid+mc*a+c;
    pid = cid+pc*a+b;

    bnormp = w[pid]*w[pid] + w[pid+d]*w[pid+d];//+w[pid+d*2]*w[pid+d*2];
    bnormm = w[mid]*w[mid] + w[mid+d]*w[mid+d];//+w[mid+d*2]*w[mid+d*2];
    c2p = gamma;
    c2m = gamma;
    d2p = 0.5*(bnormp + c2p);
    d2m = 0.5*(bnormm + c2m);

    speed_p = fabs(1.0) + sqrt(d2p + sqrt(d2p*d2p-c2p)); //sqrt(gamma*max(w[pid+d*3],P0)/max(w[pid],RHO0)));
    speed_m = fabs(1.0) + sqrt(d2m + sqrt(d2m*d2m-c2m));//sqrt(gamma*max(w[mid+d*3],P0)/max(w[mid],RHO0)));

    cmax=max(speed_m,speed_p);

    for(var = 0; var < nvar; var++){

	  FG[id+size*var]= 0.0; //0.5*(f[pid+d*var]+f[mid+d*var])-0.5*cmax*(u[pid+d*var]-u[mid+d*var]);
    //FG[id+size*var]=f[mid+d*var];

    }
  }
}


__global__ void compute_HLLC(double* u,double* w,double* FG,double gamma,int m,int ny,int nx,int nvar,int dim,int bc,int size){

  int id, cid, cell, quad, mc, pc, pid, mid, face, fsize;
  int dim1, dim2;
  double cmax, cp, cm, vp, vm, sp, sm, dp, dm, pp, pm, vstar, e;
  double wgdnv[4];
  id = blockDim.x * blockIdx.x + threadIdx.x;
  int a = nx+1-dim;
  int b = (ny+dim)*a;
  int c = m*b;
  int d;
  quad = id/b;
  if(dim == 0){
    face = id-quad*b;
    cell = face/a;
    face -= cell*a;
    cid = cell*nx + quad*nx*ny;
    fsize = nx;
    a = 1;
  }
  else if(dim == 1){
    cell = id-quad*b;
    face = cell/a;
    cell -= face*a;
    cid = cell + quad*nx*ny;
    fsize = ny;
    a = nx;
  }
  b=m*nx*ny*(2*dim); //Index for left/bottom face
  c=m*nx*ny*(2*dim+1); //Index for right/top face
  d=4*m*nx*ny;
  if(id < size){
    dim1 = dim+1;
    dim2 = 2-dim;
    mc = BC(face-1,fsize,bc);
    pc = BC(face,fsize,bc);
    pid = cid+pc*a+b; //UR=UL(face)
    mid = cid+mc*a+c; //UL=UR(face-1)
    cp = sqrt(gamma*max(w[pid+d*3],P0)/max(w[pid],RHO0));
    cm = sqrt(gamma*max(w[mid+d*3],P0)/max(w[mid],RHO0));
    cmax=max(cm,cp);
    vp = w[pid+d*dim1];
    vm = w[mid+d*dim1];
    pp = w[pid+d*3];
    pm = w[mid+d*3];
    //Compute HLL wave speed
    sm=min(vp,vm)-cmax;
    sp=max(vp,vm)+cmax;
    //Compute Lagrangian sound speed
    dm=w[mid]*(vm-sm);
    dp=w[pid]*(sp-vp);
    //Compute acoustic star state
    vstar=(dp*vp+dm*vm+(pm-pp))/(dm+dp);
    if(sm>0.0){
      wgdnv[0]=w[mid];
      wgdnv[dim1]=vm;
      wgdnv[dim2]=w[mid+d*dim2];
      wgdnv[3]=pm;
      e=u[mid+d*3];
    }
    else if(vstar>0.0){
      wgdnv[0]=w[mid]*(sm-vm)/(sm-vstar);;
      wgdnv[dim1]=vstar;
      wgdnv[dim2]=w[mid+d*dim2];
      wgdnv[3]=w[mid+d*3]+w[mid]*(sm-vm)*(vstar-vm);
      e=((sm-vm)*u[mid+d*3]-pm*vm+wgdnv[3]*vstar)/(sm-vstar);
    }
    else if(sp>0.0){
      wgdnv[0]=w[pid]*(sp-vp)/(sp-vstar);
      wgdnv[dim1]=vstar;
      wgdnv[dim2]=w[pid+d*dim2];
      wgdnv[3]=w[pid+d*3]+w[pid]*(sp-vp)*(vstar-vp);
      e=((sp-vp)*u[pid+d*3]-pp*vp+wgdnv[3]*vstar)/(sp-vstar);
    }
    else{
      wgdnv[0]=w[pid];
      wgdnv[dim1]=vp;
      wgdnv[dim2]=w[pid+d*dim2];
      wgdnv[3]=pp;
      e=u[pid+d*3];
    }
    FG[id]=wgdnv[0]*wgdnv[dim1];
    FG[id+size*dim1]=wgdnv[0]*wgdnv[dim1]*wgdnv[dim1]+wgdnv[3];
    FG[id+size*dim2]=wgdnv[0]*wgdnv[dim1]*wgdnv[dim2];
    FG[id+size*3]=wgdnv[dim1]*(e+wgdnv[3]);
  }
}

__global__ void flux_line_integral(double* edge,double* F,double* G,int m,int ny,int nx,int nvar){
  int id, ic, jc, im, jm, va;
  int q, idF, idG;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int size = nvar*d;
  double valx,valy;
  double chsi_m = -1, chsi_p = 1;
  valx=valy=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  va  = id/d;
  ic  = id - va*d;
  jm  = ic/c;
  ic -= jm*c;
  im  = ic/b;
  ic -= im*b;
  jc  = ic/a;
  ic -= jc*a;
  idF  = ic + jc*(nx+1) + va*m*ny*(nx+1);
  idG =  ic + jc*nx     + va*m*nx*(ny+1);
  b = (nx+1)*ny;
  c = (ny+1)*nx;
  if( id < size){
    for(q = 0; q < m; q++){
      valx += (F[q*b+idF+ 1]*legendre(chsi_p,im,1)-F[q*b+idF]*legendre(chsi_m,im,1))*legendre(yquad[q],jm,1)*wyquad[q];
      valy += (G[q*c+idG+nx]*legendre(chsi_p,jm,1)-G[q*c+idG]*legendre(chsi_m,jm,1))*legendre(xquad[q],im,1)*wxquad[q];
    }
    edge[id] = valx;
    edge[id+size] = valy;
  }
}

__global__ void flux_line_integral_ldf_t(double* edge,double* F,double* G, double invdx, double invdy, int m,int ny,int nx,int nvar){
  int id, ic, jc, im, jm, va;
  int q, idF, idG, idF2, idG2, mid;
  int a = nx;
  int b = ny*a;
  int c = m*b;
  int d = m*c;
  int bsi = m*(m+3)/2;
  int modek;
  //int size = nvar*d;
  int size = bsi*nx*ny;
  double valx1,valy1,valx2,valy2;
  double chsi_m = -1, chsi_p = 1;
  valx1=valy1=valx2=valy2=0.0;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  modek  = id/(nx*ny);
  ic  = id - modek*nx*ny;
  jc  = ic/a;
  ic -= jc*a;

  idF  = ic + jc*(nx+1); // + va*m*ny*(nx+1);
  idG =  ic + jc*nx; //     + va*m*nx*(ny+1);

  idF2  = ic + jc*(nx+1) + m*ny*(nx+1);
  idG2 =  ic + jc*nx + m*nx*(ny+1);

  mid = modek;

  b = (nx+1)*ny;
  c = (ny+1)*nx;
  if( id < size){
    for(q = 0; q < m; q++){
      valx1 += (F[q*b+idF+ 1]*basis_ldf_t(chsi_p,yquad[q],mid,m,0)-F[q*b+idF]*basis_ldf_t(chsi_m,yquad[q],mid,m,0))*wyquad[q];
      valy1 += (G[q*c+idG+nx]*basis_ldf_t(xquad[q],chsi_p,mid,m,0)-G[q*c+idG]*basis_ldf_t(xquad[q],chsi_m,mid,m,0))*wxquad[q];

      valx2 += (F[q*b+idF2 + 1]*basis_ldf_t(chsi_p,yquad[q],mid,m,1)-F[q*b+idF2]*basis_ldf_t(chsi_m,yquad[q],mid,m,1))*wyquad[q];
      valy2 += (G[q*c+idG2+nx]*basis_ldf_t(xquad[q],chsi_p,mid,m,1)-G[q*c+idG2]*basis_ldf_t(xquad[q],chsi_m,mid,m,1))*wxquad[q];
    }
    edge[id] = invdx*(valx1 + valx2);
    edge[id + size] = invdy*(valy2 + valy1);
  }
}

__global__ void wave_killing_bc (double* nodes, double* x, double* y, double boxlen_x, double dt, int size) {
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  double lambdadt,rho0,xn,yn,r;
  if( id < size ){
    xn = x[id];
    yn = y[id];
    r = sqrt(xn*xn+yn*yn);
    rho0= 1./(1.+pow(r/(0.3*boxlen_x),10));
    //rho0=1.0;
    lambdadt = 0.005*dt*( r < 0.4*boxlen_x ?  0.0 : pow(1.-exp(-(r-0.4*boxlen_x)/(0.04*boxlen_x)),6.0));
    nodes[id] = (nodes[id]+lambdadt*rho0)/(1+lambdadt);
    //    nodes[id] = lambdadt;
  }
}

__global__ void compute_dudt_ldf_t(double* dudt, double* flux_vol, double* edges, double invdx, double invdy, int size){
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    dudt[id] = 0.5*(flux_vol[id] - edges[id] - edges[id+size]);
  }

}

__global__ void compute_dudt(double* dudt, double* flux_vol, double* edges, double* src_vol, double invdx, double invdy, int size){
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    dudt[id] = 0.5*(flux_vol[id] - invdx*edges[id] - invdy*edges[id+size]);
  }

}

__global__ void compute_dudt_b(double* dudt, double* flux_vol, double* edges, double invdx, double invdy, int size){
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size ){
    dudt[id] = 0.5*(flux_vol[id]- invdx*edges[id] - invdy*edges[id+size]);
  }

}

__global__ void sum3 (double* out, double* A, double* B, double* C, double alpha, double beta, double gamma, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] = A[id]*alpha + B[id]*beta + C[id]*gamma;
}

__global__ void minus2 (double* out, double* A, double* B, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] = A[id] - B[id];
}

__global__ void plus_equal (double* out, double* A, double* B, double alpha, double beta, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] += A[id]*alpha + B[id]*beta;
}
__global__ void sum2 (double* out, double* A, double* B, double beta, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] = A[id] + B[id]*beta;
}

__global__ void sumequal (double* out, double* A, double* B, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] = A[id] + B[id];
}

__global__ void timesC (double* out, double* A, double constant, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] = A[id]*constant;
}

__global__ void parabolic_decay (double* out, double* A, double constant, int size)
{
  int id;
  id = blockDim.x * blockIdx.x + threadIdx.x;
  if( id < size )
    out[id] = A[id]*constant;
}

extern "C" void device_get_modes_from_nodes_(double** nodes, double** modes){
  int size = nx*ny*m*m*nvar;
  get_modes_from_nodes<<<(size+BLOCK-1)/BLOCK,BLOCK>>>(*nodes,*modes,m,ny,nx,nvar);
}


extern "C" void device_get_nodes_from_modes_(double** modes, double** nodes){
  int size = nx*ny*m*m*nvar;
  get_nodes_from_modes<<<(size+BLOCK-1)/BLOCK,BLOCK>>>(*modes,*nodes,m,ny,nx,nvar);
}

extern "C" void device_get_modes_from_nodes_ldf_b_(double** nodes, double** modes, double** bmodes){
  int bsize = m*(m+3)/2;
  get_modes_from_nodes_ldf<<<(nx*ny*bsize+BLOCK-1)/BLOCK,BLOCK>>>(*nodes,*bmodes,m,ny,nx,nvar);

}

extern "C" void device_get_nodes_from_modes_ldf_b_(double** modes, double** bmodes, double** nodes){
  int usize = nx*ny*m*m;
  get_nodes_from_modes_ldf<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(*bmodes,*nodes,m,ny,nx,nvar);
}

extern "C" void device_compute_min_dt_t_ (double* Dt){
  int size = nx*ny;
  int usize = nx*ny*m*m;
  double dt;
  // get nodes from modes
  get_nodes_from_modes_ldf<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(b_modes,w1,m,ny,nx,nvar);
  compute_primitive_t<<<(size+BLOCK-1)/BLOCK,BLOCK>>>( w1, w, gmma, m,usize, size);
  compute_min_dt<<<1,BLOCK>>>( w, pivot, gmma, cfl, dx, dy, m, usize, size);
  cudaMemcpy(&dt,pivot,sizeof(double),cudaMemcpyDeviceToHost);
  *Dt = dt;
}

extern "C" void device_compute_min_dt_ (double* Dt){
  int size = nx*ny;
  double dt;
  compute_primitive<<<(size+BLOCK-1)/BLOCK,BLOCK>>>( du, w, gmma, usize, size);
  compute_min_dt<<<1,BLOCK>>>( w, pivot, gmma, cfl, dx, dy, m, usize, size);
  cudaMemcpy(&dt,pivot,sizeof(double),cudaMemcpyDeviceToHost);
  *Dt = dt;
}

__global__ void compute_max_v(double* w, double* vm, double gamma, double cfl, double dx, double dy, int m, int usize, int size){
  int id, jump;
  double dt,dt_min,csx,csy,constant,cs,d2, v, vmax;
  __shared__ double maxs[BLOCK];
  id = threadIdx.x;


  if(id < size){
    cs = gamma;
    d2 = 0.5*((w[id]*w[id] + w[id+usize]*w[id+usize]+w[id+usize*2]*w[id+usize*2])/1.0 + cs);
    csx = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
    csy = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));

    vmax = 1.0; //sqrt(pow((fabs(1.0)),2.)+pow((fabs(1.0)),2.));
    for (id = threadIdx.x+blockDim.x; id < size; id += blockDim.x){ //This is implemented considering only one block in the reduction launch.
      cs = gamma;
      d2 = 0.5*((w[id]*w[id] + w[id+usize]*w[id+usize]+w[id+usize*2]*w[id+usize*2])/1.0 + cs);
      csx = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
      csy = sqrt(d2 + sqrt(d2*d2-cs*1.0));//sqrt(gamma*max(w[id+usize*3],P0)/max(w[id],RHO0));
      v = 1.0; //sqrt(pow((fabs(1.0)),2.)+pow((fabs(1.0)),2.));
      vmax=max(v,vmax);
    }
    maxs[threadIdx.x] = vmax;
  }
  __syncthreads();
  for(jump = blockDim.x/2; jump > 0; jump >>= 1){
    if( threadIdx.x < jump )
      maxs[threadIdx.x]=max(maxs[threadIdx.x],maxs[threadIdx.x+jump]);
    __syncthreads();
  }
  if(threadIdx.x == 0)
    vm[0] = maxs[0];
}

extern "C" void device_compute_max_v_(double *vm){
  int size = nx*ny;
  double v;
  compute_primitive<<<(size+BLOCK-1)/BLOCK,BLOCK>>>( du, w, gmma, usize, size);
  compute_max_v<<<1,BLOCK>>>( w, pivot, gmma, cfl, dx, dy, m, usize, size);
  cudaMemcpy(&v,pivot,sizeof(double),cudaMemcpyDeviceToHost);
  *vm = v;

}

extern "C" void compute_psi_correction(double* dudt, double* modes, double dt, double vm){
  double ch, cp2;
  ch = vm*5.;//cbrt(0.2*1.0/(dt*dt));
  cp2 = ch*0.18;
  // convert psi to nodes (change to only do the last variable)
  get_nodes_from_modes<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(modes,u_d_q,m,ny,nx,nvar);
  // perform integral \int psi * phi dxdy
  volume_integral<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(pivot,u_d_q,-1.0/(cp2),invdx,invdy,m,ny,nx,nvar);
  // add correction
  sum2<<< (usize+BLOCK-1)/BLOCK,BLOCK >>>(&dudt[usize*3],&dudt[usize*3], &pivot[usize*3], ch*ch, usize);

  cudaDeviceSynchronize();
}

extern "C" void device_compute_update_(int* Iter, int* SSP, double* DT, double* T, double *vm){
  double dt = *DT;
  double t = *T;
  int iter = *Iter;
  int RK = *SSP;
  double v_max = *vm;
  double* modes;
  double ch;
  
  switch (iter){
  case 0:
    modes = du;
    break;
  case 1:
    modes = w1;
    break;
  case 2:
    modes = w2;
    break;
  case 3:
    modes = w3;
    break;
  case 4:
    modes = w4;
    break;
  }

  // slve parabolic part of psi


  get_nodes_from_modes<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(modes,u_d_q,m,ny,nx,nvar);
  compute_primitive<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>( u_d_q, w, gmma, usize, usize);
  compute_flux<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(u_d_q, w, flux_q1, flux_q2, usize);

  flux_vol<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(flux_v,flux_q1,flux_q2,invdx,invdy,m,ny,nx,nvar);
  compute_faces<<<(nx*ny*nvar+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,modes,m,ny,nx,nvar);
  compute_primitive<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,gmma,4*nx*ny*m,4*nx*ny*m);

#ifdef UPWIND
  compute_flux<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,flux_f2,4*nx*ny*m);
  compute_true_upwind<<<(m*ny*(nx+1)+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,F,gmma,m,ny,nx,nvar,0,bc,m*ny*(nx+1));
  compute_true_upwind<<<(m*(ny+1)*nx+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f2,G,gmma,m,ny,nx,nvar,1,bc,m*(ny+1)*nx);
#else
    compute_flux<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,flux_f2,4*nx*ny*m);
    compute_LLF_fixed_speed<<<(m*ny*(nx+1)+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,F,gmma,m,ny,nx,nvar,0,bc,m*ny*(nx+1));
    compute_LLF_fixed_speed<<<(m*(ny+1)*nx+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f2,G,gmma,m,ny,nx,nvar,1,bc,m*(ny+1)*nx);
#endif

  flux_line_integral<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(edge,F,G,m,ny,nx,nvar);

  compute_dudt<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(dudt,flux_v,edge,src_vol,invdx,invdy,nx*ny*m*m*nvar);

  if (RK==4){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w1,du,dudt,(double)0.391752226571890*dt, tsize);
      break;
    case 1:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w2,du,w1,dudt,0.444370493651235,0.555629506348765,0.368410593050371*dt, tsize);
      break;
    case 2:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w3,du,w2,dudt,0.620101851488403,0.379898148511597,0.251891774271694*dt, tsize);
      break;
    case 3:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w4,du,w3,dudt,0.178079954393132,0.821920045606868,0.544974750228521*dt, tsize);
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,w2,w3,dudt,0.517231671970585,0.096059710526147,0.063692468666290*dt, tsize);
      break;
    case 4:
      plus_equal<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,w4,dudt,0.386708617503269,0.226007483236906*dt, tsize);
      cudaDeviceSynchronize();
      break;
    }
  }
  else if(RK==3){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w1,du,dudt,dt,tsize);
      break;
    case 1:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w2,du,w1,dudt,0.75,0.25,0.25*dt, tsize);
      break;
    case 2:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,du,w2,dudt,1.0/3.0,2.0/3.0,2.0/3.0*dt, tsize);
      cudaDeviceSynchronize();
      break;
     }
  }
  else if(RK==2){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w1,du,dudt,dt,tsize);
      break;
    case 1:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,du,w1,dudt,0.5,0.5,0.5*dt, tsize);
      cudaDeviceSynchronize();
      break;
    }
  }
  else if(RK==1){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,du,dudt,dt,tsize);
      cudaDeviceSynchronize();
      break;
    }
  }
}

extern "C" void device_compute_update_lldf_test_new_(int* Iter, int* SSP, double* DT, double* T, double *vm){
  double dt = *DT;
  double t = *T;
  int iter = *Iter;
  int RK = *SSP;
  int bs = nx*ny*m*(m+3)/2;
  double v_max = *vm;
  double* modes;
  double* bmodes;

  switch (iter){
  case 0:
    modes = du;
    bmodes = b_modes;
    break;
  case 1:
    modes = w1;
    bmodes = b_modes1;
    break;
  case 2:
    modes = w2;
    bmodes= b_modes2;
    break;
  case 3:
    modes = w3;
    bmodes= b_modes3;
    break;
  case 4:
    modes = w4;
    bmodes= b_modes4;
    break;
  }
  

  get_nodes_from_modes_ldf<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(bmodes,u_d_q,m,ny,nx,nvar); // TODO
  compute_primitive<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>( u_d_q, w, gmma, usize, usize);

  compute_flux<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(u_d_q, w, flux_q1, flux_q2, usize);

  flux_vol_ldf_t<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(flux_v_b,flux_q1,flux_q2,invdx,invdy,m,ny,nx,nvar); // TODO

  compute_faces_ldf_t<<<(nx*ny+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,bmodes,m,ny,nx,nvar); // TODO

  compute_primitive<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,gmma,4*nx*ny*m,4*nx*ny*m);

#ifdef UPWIND
  compute_flux<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,flux_f2,4*nx*ny*m);
  compute_true_upwind<<<(m*ny*(nx+1)+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,F,gmma,m,ny,nx,nvar,0,bc,m*ny*(nx+1));
  compute_true_upwind<<<(m*(ny+1)*nx+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f2,G,gmma,m,ny,nx,nvar,1,bc,m*(ny+1)*nx);
#else
    compute_flux<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,flux_f2,4*nx*ny*m);
    compute_LLF_fixed_speed<<<(m*ny*(nx+1)+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,F,gmma,m,ny,nx,nvar,0,bc,m*ny*(nx+1));
    compute_LLF_fixed_speed<<<(m*(ny+1)*nx+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f2,G,gmma,m,ny,nx,nvar,1,bc,m*(ny+1)*nx);
#endif

  flux_line_integral_ldf_t<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(edges_b,F,G,invdx,invdy,m,ny,nx,nvar); // TODO

  compute_dudt_ldf_t<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(dudt_b,flux_v_b,edges_b,invdx,invdy,bs); // TODO

  cudaDeviceSynchronize();

  if(RK==2){
    switch (iter){
    case 0:
      sum2<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes1,b_modes,dudt_b,dt,bs); // TODO
      break;
    case 1:
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes,b_modes,b_modes1,dudt_b,0.5,0.5,0.5*dt, bs); // TODO
      cudaDeviceSynchronize();
      break;
    }
  }
  else if(RK==3){
    switch (iter){
    case 0:
      sum2<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes1,b_modes,dudt_b,dt,bs);
      break;
    case 1:
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes2,b_modes,b_modes1,dudt_b,0.75,0.25,0.25*dt, bs);
      break;
    case 2:
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes,b_modes,b_modes2,dudt_b,1.0/3.0,2.0/3.0,2.0/3.0*dt, bs);
      cudaDeviceSynchronize();
      break;
     }
  }
  else if(RK==1){
    switch (iter){
    case 0:
      sum2<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes,b_modes,dudt_b,dt,bs); // TODO
      cudaDeviceSynchronize();
      break;
    }
  }
  else if(RK==4){
    switch (iter){
    case 0:
      sum2<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes1,b_modes,dudt_b,(double)0.391752226571890*dt, bs);
      break;
    case 1:
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes2,b_modes,b_modes1,dudt_b,0.444370493651235,0.555629506348765,0.368410593050371*dt, bs);
      break;
    case 2:
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes3,b_modes,b_modes2,dudt_b,0.620101851488403,0.379898148511597,0.251891774271694*dt, bs);
      break;
    case 3:
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes4,b_modes,b_modes3,dudt_b,0.178079954393132,0.821920045606868,0.544974750228521*dt, bs);
      sum3<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes,b_modes2,b_modes3,dudt_b,0.517231671970585,0.096059710526147,0.063692468666290*dt, bs);
      break;
    case 4:
      plus_equal<<<(bs+BLOCK-1)/BLOCK,BLOCK>>>(b_modes,b_modes4,dudt_b,0.386708617503269,0.226007483236906*dt, bs);
      cudaDeviceSynchronize();
      break;
    }
  }

}

extern "C" void post_process_b_(int* Iter, int* SSP, double* DT, double* T, double *vm, double *CH){
  double dt = *DT;
  double t = *T;
  int iter = *Iter;
  int RK = *SSP;
  double v_max = *vm;
  double ch = *CH;
  double dx = 1./nx;
  double* modes;

  switch (iter){
  case 0:
    modes = du;
    break;
  case 1:
    modes = w1;
    break;
  case 2:
    modes = w2;
    break;
  case 3:
    modes = w3;
    break;
  case 4:
    modes = w4;
    break;
  }

  get_nodes_from_modes<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(modes,u_d_q,m,ny,nx,nvar);
  compute_primitive<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>( u_d_q, w, gmma, usize, usize);
  compute_flux_b<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(u_d_q, w, flux_q1, flux_q2, ch, m, usize);

  flux_vol<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(flux_v,flux_q1,flux_q2,invdx,invdy,m,ny,nx,nvar);
  compute_faces<<<(nx*ny*nvar+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,modes,m,ny,nx,nvar);
  compute_primitive<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,gmma,4*nx*ny*m,4*nx*ny*m);

  compute_flux_b<<<(4*nx*ny*m+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,flux_f2, ch, m, 4*nx*ny*m);
  
  //#ifdef UPWIND
  //  compute_true_upwind<<<(m*ny*(nx+1)+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,F,gmma,m,ny,nx,nvar,0,bc,m*ny*(nx+1));
  //  compute_true_upwind<<<(m*(ny+1)*nx+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f2,G,gmma,m,ny,nx,nvar,1,bc,m*(ny+1)*nx);
  //#else
    compute_LLF_fixed_speed<<<(m*ny*(nx+1)+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f1,F,gmma,m,ny,nx,nvar,0,bc,m*ny*(nx+1));
    compute_LLF_fixed_speed<<<(m*(ny+1)*nx+BLOCK-1)/BLOCK,BLOCK>>>(ufaces,wfaces,flux_f2,G,gmma,m,ny,nx,nvar,1,bc,m*(ny+1)*nx);
  //#endif

  flux_line_integral<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(edge,F,G,m,ny,nx,nvar);

  compute_dudt_b<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(dudt,flux_v,edge,invdx,invdy,nx*ny*m*m*nvar);

  if (RK==4){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w1,du,dudt,(double)0.391752226571890*dt, tsize);
      break;
    case 1:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w2,du,w1,dudt,0.444370493651235,0.555629506348765,0.368410593050371*dt, tsize);
      break;
    case 2:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w3,du,w2,dudt,0.620101851488403,0.379898148511597,0.251891774271694*dt, tsize);
      break;
    case 3:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w4,du,w3,dudt,0.178079954393132,0.821920045606868,0.544974750228521*dt, tsize);
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,w2,w3,dudt,0.517231671970585,0.096059710526147,0.063692468666290*dt, tsize);
      break;
    case 4:
      plus_equal<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,w4,dudt,0.386708617503269,0.226007483236906*dt, tsize);
      cudaDeviceSynchronize();
      break;
    }
  }
  else if(RK==3){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w1,du,dudt,dt,tsize);
      break;
    case 1:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w2,du,w1,dudt,0.75,0.25,0.25*dt, tsize);
      break;
    case 2:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,du,w2,dudt,1.0/3.0,2.0/3.0,2.0/3.0*dt, tsize);
      cudaDeviceSynchronize();
      break;
     }
  }
  else if(RK==2){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(w1,du,dudt,dt,tsize);
      break;
    case 1:
      sum3<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,du,w1,dudt,0.5,0.5,0.5*dt, tsize);
      cudaDeviceSynchronize();
      break;
    }
  }
  else if(RK==1){
    switch (iter){
    case 0:
      sum2<<<(tsize+BLOCK-1)/BLOCK,BLOCK>>>(du,du,dudt,dt,tsize);
      cudaDeviceSynchronize();
      break;
    }
  }
}

extern "C" void parabolic_psi_(double* DT, double* T, double *vm, double *Dx, double *Coeff){
  double dt = *DT;
  double t = *T;
  double dx = *Dx;
  double v_max = *vm;
  double* modes;
  double constant;
  double coeff = *Coeff;
  modes = du;
  constant = exp(-coeff);
  parabolic_decay<<<(usize+BLOCK-1)/BLOCK,BLOCK>>>(&modes[usize*3],&modes[usize*3],constant,usize);
  cudaDeviceSynchronize();
}

extern "C" void mem_check(size_t free, size_t total){
  //size_t free;
  //size_t total;
  cudaError_t error = cudaMemGetInfo(&free,&total);
  cudaError_t error2 = cudaGetLastError();
  printf("free: %ld, total: %ld \n",free, total);
  if(error2 != cudaSuccess){
    printf("Mem_check_error: %s\n", cudaGetErrorString(error2));
  }

}

template < typename T >
inline void __checkCudaErrors(T result, char const *const func, const char *const file, int const line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s:%i : checkCudaErrors() CUDA error (#%d): %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

#define CCE(val) __checkCudaErrors( (val), #val, __FILE__, __LINE__ )


extern "C" void gpu_allocation_ (int *Nvar, int* Nx, int* Ny, int* M, int* K, double *Bl_x, double *Bl_y, double *CFL, double *Eta, int* Bc, double *Gamma) {
  size_t free;
  size_t total;
  nvar = *Nvar;
  nx = *Nx;
  ny = *Ny;
  m = *M;
  k = *K;
  bc = *Bc;
  usize = m*m*nx*ny;
  tsize = usize*nvar;
  boxlen_x = *Bl_x;
  boxlen_y = *Bl_y;
  dx = boxlen_x/double(nx);
  dy = boxlen_y/double(ny);
  invdx = 1/dx;
  invdy = 1/dy;
  cfl = *CFL;
  gmma = *Gamma;
  eta = *Eta;
  vm = 0;
  bsize = m*(m+3)/2;
  //cudaError_t error = cudaGetLastError();
  //cudaError_t memerror = cudaMemGetInfo(&free,&total);
  //error = cudaGetLastError();
  cudaMalloc ( &u, tsize * sizeof(double));
  cudaMalloc ( &u_d_q, tsize * sizeof(double));
  cudaMalloc ( &du, tsize * sizeof(double));
  cudaMalloc ( &w, tsize * sizeof(double));
  cudaMalloc ( &w1, tsize * sizeof(double));
  cudaMalloc ( &w2, tsize * sizeof(double));
  cudaMalloc ( &w3, tsize * sizeof(double));
  cudaMalloc ( &w4, tsize * sizeof(double));
  cudaMalloc ( &dudt, tsize * sizeof(double));

  cudaMalloc ( &b_modes, bsize*nx*ny * sizeof(double));
  cudaMalloc ( &b_modes1, bsize*m*nx*ny * sizeof(double));
  cudaMalloc ( &b_modes2, bsize*m*nx*ny * sizeof(double));
  cudaMalloc ( &b_modes3, bsize*m*nx*ny * sizeof(double));
  cudaMalloc ( &b_modes4, bsize*m*nx*ny * sizeof(double));

  cudaMalloc ( &flux_v_b,  bsize*nx*ny * sizeof(double));
  cudaMalloc ( &edges_b, 2*bsize*nx*ny * sizeof(double));
  cudaMalloc ( &dudt_b, bsize*nx*ny * sizeof(double));

  cudaMalloc ( &ufaces, 4*nvar*nx*ny*m*sizeof(double));
  cudaMalloc ( &wfaces, 4*nvar*nx*ny*m*sizeof(double));
  cudaMalloc ( &flux_f1, 4*nvar*nx*ny*m*sizeof(double));
  cudaMalloc ( &flux_f2, 4*nvar*nx*ny*m*sizeof(double));
  cudaMalloc ( &flux_q1, tsize * sizeof(double));
  cudaMalloc ( &flux_q2, tsize * sizeof(double));
  cudaMalloc ( &flux_v,  tsize * sizeof(double));
  cudaMalloc ( &F, nvar*(nx+1)*ny*m * sizeof(double));
  cudaMalloc ( &G, nvar*nx*(ny+1)*m * sizeof(double));
  cudaMalloc ( &edge, tsize*2 * sizeof(double));
  /*error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("Error1");
    printf("CUDA error gpu init: %s\n", cudaGetErrorString(error));
    exit(-1);
  }*/

  cudaMalloc ( &x, nx*ny*m*m * sizeof(double));
  cudaMalloc ( &y, nx*ny*m*m * sizeof(double));
  cudaMalloc ( &uX, nvar*ny*nx*k*m*sizeof(double));
  cudaMalloc ( &uY, nvar*ny*nx*k*m*sizeof(double));
  cudaMalloc ( &pivot, tsize * sizeof(double));
  cudaMalloc ( &xc, nx*ny * sizeof(double));
  cudaMalloc ( &yc, nx*ny * sizeof(double));
  cudaMalloc ( &pivot1, tsize * sizeof(double));

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("Allocated all, CUDA error gpu init: %s\n", cudaGetErrorString(error));
    exit(-1);
    mem_check(free,total);
  }
  printf("GPU allocation done\n");
}

extern "C" void gpu_set_pointers_ (double** u_d, double** du_d, double** dudt_d, double** w_d,
			       double** x_d, double** y_d , double** xc_d, double** yc_d, double* x_quad, double* y_quad, double* w_x_quad,
				   double* w_y_quad, double* x_gll, double* y_gll, double* w_x_gll,
				   double* w_y_gll, double* Sqrt_mod) {
  cudaError_t error = cudaGetLastError();
  *u_d = u;
  *du_d =du;
  *dudt_d =dudt;
  *w_d = w;
  *x_d = x;
  *y_d = y;
  *xc_d = xc;
  *yc_d = yc;
  cudaMemcpyToSymbol(sqrt_mod,Sqrt_mod,sizeof(double)*m);
  cudaMemcpyToSymbol(xquad,x_quad,sizeof(double)*m);
  cudaMemcpyToSymbol(yquad,y_quad,sizeof(double)*m);
  cudaMemcpyToSymbol(wxquad,w_x_quad,sizeof(double)*m);
  cudaMemcpyToSymbol(wyquad,w_y_quad,sizeof(double)*m);
  cudaMemcpyToSymbol(xgll,x_gll,sizeof(double)*k);
  cudaMemcpyToSymbol(ygll,y_gll,sizeof(double)*k);
  cudaMemcpyToSymbol(wxgll,w_x_gll,sizeof(double)*k);
  cudaMemcpyToSymbol(wygll,w_y_gll,sizeof(double)*k);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("CUDA error gpu init: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  printf("FORTRAN-CUDA pointers done\n");
}

extern "C" void gpu_set_more_pointers_ (double* Sqrt_div, double** b_d) {
  cudaError_t error = cudaGetLastError();
  *b_d = b_modes;
  cudaMemcpyToSymbol(sqrts_div,Sqrt_div,sizeof(double)*m);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("CUDA error gpu init: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  printf("FORTRAN-CUDA pointers done\n");
}



extern "C" void h2d_ (double *array, double **darray, int* Size) {
  int size = *Size;
  cudaError_t error = cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy( *darray, array,  size * sizeof(double) ,cudaMemcpyHostToDevice);
  error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("CUDA error h2d: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

extern "C" void d2h_ (double **darray, double *array, int* Size) {
  int size = *Size;
  cudaError_t error = cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy( array, *darray,  size * sizeof(double) ,cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("CUDA error d2h: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

extern "C" void setdevice_ (int *Device) {
  int device = *Device;
  cudaSetDevice(device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("Device Number: %d\n", device);
  printf("Device name: %s\n", prop.name);
  printf("Device Memory: %lu\n",prop.totalGlobalMem);
}

extern "C" void devices_ () {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  printf("Devices: %d\n",nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Name: %s\n", prop.name);
    printf("  Compute mode: %d\n", prop.computeMode);
    printf("  Memory Capacity (bytes): %lu\n",
           prop.totalGlobalMem);
    printf("  Multiprocessors: %d\n\n", prop.multiProcessorCount);
  }
}
