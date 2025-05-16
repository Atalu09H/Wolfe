import math
import sys
import numpy as np

# Define constants
INT_INF = sys.maxsize
INF = sys.float_info.max

FALSE = False
TRUE = True
NULL = None

# cg_parameter structure
class CGParameter:
    def __init__(self):
        self.PrintFinal = False
        self.PrintLevel = 0
        self.PrintParms = False
        self.AWolfe = False
        self.AWolfeFac = 1.e-3
        self.Qdecay = 0.7
        self.nslow = 1000
        self.StopRule = True
        self.StopFac = 0.e-12
        self.PertRule = True
        self.eps = 1.e-6
        self.egrow = 10.0
        self.QuadStep = True
        self.QuadCutOff = 1.e-12
        self.QuadSafe = 1.e-3
        self.UseCubic = True
        self.CubicCutOff = 1.e-12
        self.SmallCost = 1.e-30
        self.debug = False
        self.debugtol = 1.e-10
        self.step = 0.0
        self.maxit_fac = INF
        self.nexpand = 50
        self.ExpandSafe = 200.0
        self.SecantAmp = 1.05
        self.RhoGrow = 2.0
        self.neps = 5
        self.nshrink = 10
        self.nline = 50
        self.restart_fac = 6.0
        self.feps = 0.0
        self.nan_rho = 1.3
        self.nan_decay = 0.1
        self.delta = 0.1
        self.sigma = 0.9
        self.gamma = 0.66
        self.rho = 5.0
        self.psi0 = 0.01
        self.psi1 = 0.1
        self.psi2 = 2.0
        self.AdaptiveBeta = False
        self.BetaLower = 0.4
        self.theta = 1.0
        self.qeps = 1.e-12
        self.qrestart = 3
        self.qrule = 1.e-8


# cg_stats structure
class CGStats:
    def __init__(self):
        self.f = None 
        self.gnorm = None 
        self.nfunc = None 
        self.ngrad = None 
        self.g = None 
        self.alpha = None 
