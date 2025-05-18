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
        self.n = 0  # problem dimension, saved for reference
        self.nf = 0  # number of function evaluations
        self.ng = 0  # number of gradient evaluations
        self.QuadOK = False  # T (quadratic step successful)
        self.UseCubic = False  # Default to secant step
        self.neps = 0  # number of times eps updated
        self.PertRule = False  # T => estimated error in function value is eps*Ck
        self.QuadF = False  # T => function appears to be quadratic
        self.SmallCost = 1e-30  # Small cost threshold
        self.alpha = 1.0  # Default step size
        self.f = 0.0  # Function value for step alpha
        self.df = 0.0  # Function derivative for step alpha
        self.fpert = 0.0  # Perturbation is eps*|f| if PertRule is T
        self.eps = 1e-6  # Default value of eps
        self.tol = 1e-8  # Default computing tolerance
        self.f0 = 0.0  # Old function value
        self.df0 = 0.0  # Old derivative
        self.Ck = 0.0  # Average cost
        self.wolfe_hi = 0.0  # Upper bound for slope in Wolfe test
        self.wolfe_lo = 0.0  # Lower bound for slope in Wolfe test
        self.awolfe_hi = 0.0  # Upper bound for slope, approximate Wolfe test
        self.AWolfe = False  # F (use Wolfe line search)
        self.Wolfe = False  # T (means code reached the Wolfe part of cg_line)
        self.rho = 1.0  # Default value for rho
        self.x = np.zeros(1)  # Current iterate (initialize as zero vector)
        self.xtemp = np.zeros(1)  # x + alpha*d (initialize as zero vector)
        self.d = np.zeros(1)  # Current search direction (initialize as zero vector)
        self.g = np.zeros(1)  # Gradient at x (initialize as zero vector)
        self.gtemp = np.zeros(1)  # Gradient at x + alpha*d (initialize as zero vector)
        self.cg_value = None  # Function for evaluating f
        self.cg_grad = None  # Function for evaluating gradient
        self.cg_valgrad = None  # Function for evaluating both f and gradient
        self.Parm = lsu.CGParameter()  # User parameters