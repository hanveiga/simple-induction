//global cpu variables
int nvar;
int nx;
int ny;
int m;
int k;
int bc;
int tsize;
int usize;
int neql; /// wellbalanced ic
int bsize;
double boxlen_x;
double boxlen_y;
double dx;
double dy;
double invdx;
double invdy;
double cfl;
double gmma;
double eta;
double dt_grav;
double vm;
//global gpu variables
double *u;
double *du;
double *w;
double *u_eq;
double *u_d_q;
double *w1;
double *w2;
double *w3;
double *w4;
double *w5;
double *dudt;
double *x;
double *y;
double *flux_v;
double *flux_q1;
double *flux_q2;
double *ul;
double *ur;
double *ub;
double *ut;
double *wl;
double *wr;
double *wb;
double *wt;
double *ufaces;
double *wfaces;
double *flux_f1;
double *flux_f2;
double *F;
double *G;
double *edge;
double *src;
double *src_vol;
double *grad;
double *phi;
double *pivot;
double *uY;
double *uX;
// well balanced
double *edge_eq;
double *w_eq;
double *modes_eq;
double *xc;
double *yc;
double *pivot1;
double *ufaces_eq;
double *ufaces_pert;
//mhd
double *b_modes;
double *b_modes1;
double *b_modes2;
double *b_modes3;
double *b_modes4;
double *flux_v_b;
double *dudt_b;
double *edges_b;
