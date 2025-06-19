#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define INT long int
#define INT_INF LONG_MAX
#define INF DBL_MAX

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef NULL
#define NULL 0
#endif

/*============================================================================
   cg_parameter is a structure containing parameters used in cg_descent
   cg_default assigns default values to these parameters */
typedef struct cg_parameter_struct /* user controlled parameters */
{
/*============================================================================
      parameters that the user may wish to modify
  ----------------------------------------------------------------------------*/
    /* T => print final statistics
       F => no printout of statistics */
    int PrintFinal ;

    /* Level 0  = no printing), ... , Level 3 = maximum printing */
    int PrintLevel ;

    /* T => print parameters values
       F => do not display parmeter values */
    int PrintParms ;

    /* T => use approximate Wolfe line search
       F => use ordinary Wolfe line search, switch to approximate Wolfe when
                |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost  */
    int    AWolfe ;
    double AWolfeFac ;

    /* factor in [0, 1] used to compute average cost magnitude C_k as follows:
       Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k */
    double Qdecay ;

    /* terminate after nslow iterations without strict improvement in
       either function value or gradient */
    int nslow ;

    /* Stop Rules:
       T => ||proj_grad||_infty <= max(grad_tol,initial ||grad||_infty*StopFact)
       F => ||proj_grad||_infty <= grad_tol*(1 + |f_k|) */
    int    StopRule ;
    double StopFac ;

    /* T => estimated error in function value is eps*Ck,
       F => estimated error in function value is eps */
    int    PertRule ;
    double eps ;

    /* factor by which eps grows when line search fails during contraction */
    double egrow ;

    /* T => attempt quadratic interpolation in line search when
                |f_k+1 - f_k|/f_k <= QuadCutoff
       F => no quadratic interpolation step */
    int    QuadStep ;
    double QuadCutOff ;

    /* maximum factor by which a quad step can reduce the step size */
    double QuadSafe ;

    /* T => when possible, use a cubic step in the line search */
    int UseCubic ;

    /* use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff */
    double CubicCutOff ;

    /* |f| < SmallCost*starting cost => skip QuadStep and set PertRule = FALSE*/
    double SmallCost ;

    /* T => check that f_k+1 - f_k <= debugtol*C_k
       F => no checking of function values */
    int    debug ;
    double debugtol ;

    /* if step is nonzero, it is the initial step of the initial line search */
    double step ;

    /* abort cg after maxit_fac*n iterations */
    double maxit_fac ;

    /* maximum number of times the bracketing interval grows during expansion */
    int nexpand ;

    /* maximum factor secant step increases stepsize in expansion phase */
    double ExpandSafe ;

    /* factor by which secant step is amplified during expansion phase
       where minimizer is bracketed */
    double SecantAmp ;

    /* factor by which rho grows during expansion phase where minimizer is
       bracketed */
    double RhoGrow ;

    /* maximum number of times that eps is updated */
    int neps ;

    /* maximum number of times the bracketing interval shrinks */
    int nshrink ;

   /* maximum number of iterations in line search */
    int nline ;

    /* conjugate gradient method restarts after (n*restart_fac) iterations */
    double restart_fac ;

    /* stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|*/
    double feps ;

    /* after encountering nan, growth factor when searching for
       a bracketing interval */
    double nan_rho ;

    /* after encountering nan, decay factor for stepsize */
    double nan_decay ;

/*============================================================================
       technical parameters which the user probably should not touch
  ----------------------------------------------------------------------------*/
    double           delta ; /* Wolfe line search parameter */
    double           sigma ; /* Wolfe line search parameter */
    double           gamma ; /* decay factor for bracket interval width */
    double             rho ; /* growth factor when searching for initial
                                bracketing interval */
    double            psi0 ; /* factor used in starting guess for iteration 1 */
    double            psi1 ; /* in performing a QuadStep, we evaluate the
                                function at psi1*previous step */
    double            psi2 ; /* when starting a new cg iteration, our initial
                                guess for the line search stepsize is
                                psi2*previous step */
    int       AdaptiveBeta ; /* T => choose beta adaptively, F => use theta */
    double       BetaLower ; /* lower bound factor for beta */
    double           theta ; /* parameter describing the cg_descent family */
    double            qeps ; /* parameter in cost error for quadratic restart
                                criterion */
    double           qrule ; /* parameter used to decide if cost is quadratic */
    int           qrestart ; /* number of iterations the function should be
                                nearly quadratic before a restart */
} cg_parameter ;

typedef struct cg_stats_struct /* statistics returned to user */
{
    double               f ; /*function value at solution */
    double           gnorm ; /* max abs component of gradient */
    INT              nfunc ; /* number of function evaluations */
    INT              ngrad ; /* number of gradient evaluations */
	double               *g; /* gradient at solution */
	double            alpha; /* x_{k+1} = x_k + \alpha * d */
} cg_stats ;

/* prototypes */
int line_search /*  return:
				   -2 (function value became nan)
				   -1 (starting function value is nan)
				   0 (convergence tolerance satisfied)
				   1 (change in func <= feps*|f|)
				   2 (total iterations exceeded maxit)
				   3 (slope always negative in line search)
				   4 (number of line search iterations exceeds nline)
				   5 (search direction not a descent direction)
				   6 (excessive updating of eps)
				   7 (Wolfe conditions never satisfied)
				   8 (debugger is on and the function value increases)
				   9 (out of memory)
				   10 (function becomes nan)
				   11 (no cost or gradient improvement in
				   2n + Parm->nslow iterations)*/
				   (
				   double            *x, /* input: starting guess, output: the solution */
				   INT                n, /* problem dimension */
				   double            *dir, /* direction */
				   cg_stats       *Stat, /* structure with statistics (can be NULL) */
				   cg_parameter  *UParm, /* user parameters, NULL = use default parameters */
				   double      (*value) (double *, INT),  /* f = value (x, n) */
				   void         (*grad) (double *, double *, INT), /* grad (g, x, n) */
				   double    (*valgrad) (double *, double *, INT) /* f = valgrad (g, x, n)*/
				   );

void cg_default /* set default parameter values */
(
    cg_parameter   *Parm
) ;
