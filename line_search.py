import math
import sys
import numpy as np
import line_search_user as lsu

# Define constants

INT_INF = sys.maxsize
INF = sys.float_info.max


# cg_com structure
class CGCom:
    def __init__(self):
        self.n = 0 # problem dimension, saved for reference
        self.nf = 0 # number of function evaluations
        self.ng = 0  # number of gradient evaluations
        self.QuadOK = False   # T (quadratic step successful)
        self.UseCubic = None   # T (use cubic step) F (use secant step)
        self.neps = 0   # number of times eps updated
        self.PertRule = False   # T => estimated error in function value is eps*Ck
        self.QuadF = False   # T => function appears to be quadratic
        self.SmallCost = 0   # |f| <= SmallCost => set PertRule = F
        self.alpha = 0.0   # stepsize along search direction
        self.f =  0.0   # function value for step alpha
        self.df =  0.0   # function derivative for step alpha
        self.fpert =  0.0   # perturbation is eps*|f| if PertRule is T
        self.eps =  0.0   # current value of eps
        self.tol = 0.0   # computing tolerance
        self.f0 =  0.0   # old function value
        self.df0 =  0.0   # old derivative
        self.Ck =  0.0   # average cost
        self.wolfe_hi =  0.0   # upper bound for slope in Wolfe test
        self.wolfe_lo =  0.0   # lower bound for slope in Wolfe test
        self.awolfe_hi =  0.0   # upper bound for slope, approximate Wolfe test
        self.AWolfe = False   # F (use Wolfe line search)
        self.Wolfe = False   # T (means code reached the Wolfe part of cg_line)
        self.rho =  0.0   # either Parm->rho or Parm->nan_rho
        self.x = None   # current iterate
        self.xtemp = None   # x + alpha*d
        self.d = None   # current search direction
        self.g = None   # gradient at x
        self.gtemp = None   # gradient at x + alpha*d
        self.cg_value = None   # function for evaluating f
        self.cg_grad = None   # function for evaluating gradient
        self.cg_valgrad = None   # function for evaluating both f and gradient
        self.Parm = lsu.CGParameter()  # user parameters