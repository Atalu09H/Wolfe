import math
import sys
import numpy as np
import line_search_user as lsu
import gradient_method as grm

# Define constants

INT_INF = sys.maxsize
INF = sys.float_info.max


# cg_com structure
class CGCom:
    def __init__(self):
        self.n = None  # problem dimension, saved for reference
        self.nf = None  # number of function evaluations
        self.ng = None  # number of gradient evaluations
        self.QuadOK = None  # T (quadratic step successful)
        self.UseCubic = None  # Default to secant step
        self.neps = None  # number of times eps updated
        self.PertRule = None  # T => estimated error in function value is eps*Ck
        self.QuadF = None  # T => function appears to be quadratic
        self.SmallCost = None  # Small cost threshold
        self.alpha = None  # Default step size
        self.f = None  # Function value for step alpha
        self.df = None  # Function derivative for step alpha
        self.fpert = None  # Perturbation is eps*|f| if PertRule is T
        self.eps = None  # Default value of eps
        self.tol = 1e-8  # Default computing tolerance
        self.f0 = None  # Old function value
        self.df0 = None  # Old derivative
        self.Ck = None  # Average cost
        self.wolfe_hi = None  # Upper bound for slope in Wolfe test
        self.wolfe_lo = None  # Lower bound for slope in Wolfe test
        self.awolfe_hi = None  # Upper bound for slope, approximate Wolfe test
        self.AWolfe = None  # F (use Wolfe line search)
        self.Wolfe = None  # T (means code reached the Wolfe part of cg_line)
        self.rho = None  # Default value for rho
        self.x = None  # Current iterate (initialize as zero vector)
        self.xtemp = None  # x + alpha*d (initialize as zero vector)
        self.d = None  # Current search direction (initialize as zero vector)
        self.g = None  # Gradient at x (initialize as zero vector)
        self.gtemp = None  # Gradient at x + alpha*d (initialize as zero vector)
        self.cg_value = None  # Function for evaluating f
        self.cg_grad = None  # Function for evaluating gradient
        self.cg_valgrad = None  # Function for evaluating both f and gradient
        self.Parm = None  # User parameters