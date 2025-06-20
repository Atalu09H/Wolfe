import numpy as np
import math
import sys
import line_search_user as lsu
import line_search as ls


# Com = ls.CGCom()
# Parm = lsu.CGParameter()
# Parm.print_Parms()
# False = False
# True = True

INT_INF = sys.maxsize
INF = sys.float_info.max


def cg_default(Parm):
    
    # Parm = lsu.CGParameter()

    Parm.PrintFinal = False
    Parm.PrintLevel = 0
    Parm.PrintParms = False
    Parm.AWolfe = False
    Parm.AWolfeFac = 1.e-3
    Parm.Qdecay = 0.7
    Parm.nslow = 1000
    Parm.StopRule = True
    Parm.StopFac = 0.e-12
    Parm.PertRule = True
    Parm.eps = 1.e-6
    Parm.egrow = 10.0
    Parm.QuadStep = True
    Parm.QuadCutOff = 1.e-12
    Parm.QuadSafe = 1.e-3
    Parm.UseCubic = True
    Parm.CubicCutOff = 1.e-12
    Parm.SmallCost = 1.e-30
    Parm.debug = False
    Parm.debugtol = 1.e-10
    Parm.step = 0.0
    Parm.maxit_fac = INF
    Parm.nexpand = 50
    Parm.ExpandSafe = 200.0
    Parm.SecantAmp = 1.05
    Parm.RhoGrow = 2.0
    Parm.neps = 5
    Parm.nshrink = 10
    Parm.nline = 50
    Parm.restart_fac = 6.0
    Parm.feps = 0.0
    Parm.nan_rho = 1.3
    Parm.nan_decay = 0.1
    Parm.delta = 0.1
    Parm.sigma = 0.9
    Parm.gamma = 0.66
    Parm.rho = 5.0
    Parm.psi0 = 0.01
    Parm.psi1 = 0.1
    Parm.psi2 = 2.0
    Parm.AdaptiveBeta = False
    Parm.BetaLower = 0.4
    Parm.theta = 1.0
    Parm.qeps = 1.e-12
    Parm.qrestart = 3
    Parm.qrule = 1.e-8

    # return Parm

########################################################################

def cg_printParms(Parm):
    print("PARAMETERS:\n")
    print(f"\nWolfe line search parameter ..................... delta: {Parm.delta:e}")
    print(f"Wolfe line search parameter ..................... sigma: {Parm.sigma:e}")
    print(f"decay factor for bracketing interval ............ gamma: {Parm.gamma:e}")
    print(f"growth factor for bracket interval ................ rho: {Parm.rho:e}")
    print(f"growth factor for bracket interval after nan .. nan_rho: {Parm.nan_rho:e}")
    print(f"decay factor for stepsize after nan ......... nan_decay: {Parm.nan_decay:e}")
    print(f"parameter in lower bound for beta ........... BetaLower: {Parm.BetaLower:e}")
    print(f"parameter describing cg_descent family .......... theta: {Parm.theta:e}")
    print(f"perturbation parameter for function value ......... eps: {Parm.eps:e}")
    print(f"factor by which eps grows if necessary .......... egrow: {Parm.egrow:e}")
    print(f"factor for Computing average cost .............. Qdecay: {Parm.Qdecay:e}")
    print(f"relative change in cost to stop quadstep ... QuadCutOff: {Parm.QuadCutOff:e}")
    print(f"maximum factor quadstep reduces stepsize ..... QuadSafe: {Parm.QuadSafe:e}")
    print(f"skip quadstep if |f| <= SmallCost*start cost  SmallCost: {Parm.SmallCost:e}")
    print(f"relative change in cost to stop cubic step  CubicCutOff: {Parm.CubicCutOff:e}")
    print(f"terminate if no improvement over nslow iter ..... nslow: {Parm.nslow}")
    print(f"factor multiplying gradient in stop condition . StopFac: {Parm.StopFac:e}")
    print(f"cost change factor, approx Wolfe transition . AWolfeFac: {Parm.AWolfeFac:e}")
    print(f"restart cg every restart_fac*n iterations . restart_fac: {Parm.restart_fac:e}")
    print(f"cost error in quadratic restart is qeps*cost ..... qeps: {Parm.qeps:e}")
    print(f"number of quadratic iterations before restart  qrestart: {Parm.qrestart}")
    print(f"parameter used to decide if cost is quadratic ... qrule: {Parm.qrule:e}")
    print(f"stop when cost change <= feps*|f| ................ feps: {Parm.feps:e}")
    print(f"starting guess parameter in first iteration ...... psi0: {Parm.psi0:e}")
    print(f"starting step in first iteration if nonzero ...... step: {Parm.step:e}")
    print(f"factor multiply starting guess in quad step ...... psi1: {Parm.psi1:e}")
    print(f"initial guess factor for general iteration ....... psi2: {Parm.psi2:e}")
    print(f"max iterations is n*maxit_fac ............... maxit_fac: {Parm.maxit_fac:e}")
    print(f"max number of contracts in the line search .... nshrink: {Parm.nshrink}")
    print(f"max expansions in line search ................. nexpand: {Parm.nexpand}")
    print(f"maximum growth of secant step in expansion . ExpandSafe: {Parm.ExpandSafe:e}")
    print(f"growth factor for secant step during expand . SecantAmp: {Parm.SecantAmp:e}")
    print(f"growth factor for rho during expansion phase .. RhoGrow: {Parm.RhoGrow:e}")
    print(f"max number of times that eps is updated .......... neps: {Parm.neps}")
    print(f"max number of iterations in line search ......... nline: {Parm.nline}")
    print(f"print level (0 = none, 2 = maximum) ........ PrintLevel: {Parm.PrintLevel}")

    # Logical parameters
    print("Logical parameters:")
    if Parm.PertRule:
        print("    Error estimate for function value is eps*Ck")
    else:
        print("    Error estimate for function value is eps")
    if Parm.QuadStep:
        print("    Use quadratic interpolation step")
    else:
        print("    No quadratic interpolation step")
    if Parm.UseCubic:
        print("    Use cubic interpolation step when possible")
    else:
        print("    Avoid cubic interpolation steps")
    if Parm.AdaptiveBeta:
        print("    Adaptively adjust direction update parameter beta")
    else:
        print("    Use fixed parameter theta in direction update")
    if Parm.PrintFinal:
        print("    Print final cost and statistics")
    else:
        print("    Do not print final cost and statistics")
    if Parm.PrintParms:
        print("    Print the parameter structure")
    else:
        print("    Do not print parameter structure")
    if Parm.AWolfe:
        print("    Approximate Wolfe line search")
    else:
        print("    Wolfe line search")
    if Parm.AWolfeFac > 0.0:
        print(" ... switching to approximate Wolfe")
    else:
        print()
    if Parm.StopRule:
        print("    Stopping condition uses initial grad tolerance")
    else:
        print("    Stopping condition weighted by absolute cost")
    if Parm.debug:
        print("    Check for decay of cost, debugger is on")
    else:
        print("    Do not check for decay of cost, debugger is off")

#######################################################################

def cg_dot(x, y, n):
    t = 0.0
    n5 = n % 5
    for i in range(n5):
        t += x[i] * y[i]
    for i in range(n5, n, 5):
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2] + x [i+3]*y [i+3] + x [i+4]*y [i+4]
        # t += np.dot(x[i:i+5], y[i:i+5])
    return t

# def cg_dot(x, y, n)-> float:
#     return np.dot(x[:n], y[:n])

#######################################################################

def cg_copy(y, x, n):
    # np.copyto(y[:n], x[:n])
    n10 = n % 10 ;
    for j in range(n10):
        y [j] = x [j] 
    for j in range(n10, n, 10):
        y [j] = x [j] 
        y [j+1] = x [j+1] 
        y [j+2] = x [j+2] 
        y [j+3] = x [j+3] 
        y [j+4] = x [j+4] 
        y [j+5] = x [j+5] 
        y [j+6] = x [j+6] 
        y [j+7] = x [j+7] 
        y [j+8] = x [j+8] 
        y [j+9] = x [j+9]
#######################################################################

def cg_step(xtemp, x, d, alpha, n):
    # np.add(x[:n], alpha * d[:n], out=xtemp[:n])
    n5 = n % 5
    for i in range(n5):
        xtemp [i] = x[i] + alpha*d[i] 
        
    for i in range(n5, n, 5):
        xtemp [i]   = x [i]   + alpha*d [i] 
        xtemp [i+1] = x [i+1] + alpha*d [i+1] 
        xtemp [i+2] = x [i+2] + alpha*d [i+2] 
        xtemp [i+3] = x [i+3] + alpha*d [i+3] 
        xtemp [i+4] = x [i+4] + alpha*d [i+4] 

#######################################################################

# Com = ls.CGCom

def cg_tol(gnorm, Com):
    if Com.Parm.StopRule:
        if gnorm <= Com.tol:
            return 1
    elif gnorm <= Com.tol * (1.0 + abs(Com.f)): return 1
    return 0
#######################################################################


def cg_cubic(a, fa, da, b, fb, db):
    delta = b - a
    if delta == 0.0:
        return a
    
    v = da + db - 3.0 * (fb - fa) / delta
    t = v * v - da * db
    
    if t < 0.0:  # Complex roots, use secant method
        if abs(da) < abs(db):
            c = a - (a - b) * (da / (da - db))
        elif da != db:
            c = b - (a - b) * (db / (da - db))
        else:
            c = -1
        return c
    
    w = math.sqrt(t) if delta > 0.0 else -math.sqrt(t)
    d1 = da + v - w
    d2 = db + v + w
    
    if d1 == 0.0 and d2 == 0.0:
        return -1.0
    
    c = a + delta * da / d1 if abs(d1) >= abs(d2) else b - delta * db / d2
    return c

######################################################################
        
# Com = ls.CGCom

def cg_evaluate(what, nan, Com):
    # Parm = lsu.CGParameter()
    Parm = Com.Parm
    
    n = Com.n
    x = Com.x
    d = Com.d
    xtemp = Com.xtemp
    gtemp = Com.gtemp
    alpha = Com.alpha

    # Check if nan should be checked
    if nan == "y":
        if what == "f":  # Compute function only
            cg_step(xtemp, x, d, alpha, n)
            Com.f = Com.cg_value(xtemp, n)
            Com.nf += 1

            # Reduce step size if function value is NaN or INF
            if (not np.isnan(Com.f)) or Com.f >= INF:
                for i in range(Parm.nexpand):
                    alpha *= Parm.nan_decay
                    cg_step(xtemp, x, d, alpha, n)
                    Com.f = Com.cg_value(xtemp, n)
                    Com.nf += 1
                    if np.isnan(Com.f) and Com.f < INF:
                        break
                if i == Parm.nexpand:
                    return -2
            Com.alpha = alpha
        elif what == "g":  # Compute gradient only
            cg_step(xtemp, x, d, alpha, n)
            Com.cg_grad(gtemp, xtemp, n)
            Com.ng += 1
            Com.df = cg_dot(gtemp, d, n)

            # Reduce step size if derivative is NaN or INF
            if (not np.isnan(Com.df)) or Com.df >= INF:
                for i in range(Parm.nexpand):
                    alpha *= Parm.nan_decay
                    cg_step(xtemp, x, d, alpha, n)
                    Com.cg_grad(gtemp, xtemp, n)
                    Com.ng += 1
                    Com.df = cg_dot(gtemp, d, n)
                    if np.isnan(Com.df) and Com.df < INF:
                        break
                if i == Parm.nexpand:
                    return -2
                Com.rho = Parm.nan_rho
            else:
                Com.rho = Parm.rho
            Com.alpha = alpha
        else:  # Compute both function and gradient
            cg_step(xtemp, x, d, alpha, n)
            if Com.cg_valgrad != None:
                Com.f = Com.cg_valgrad(gtemp, xtemp, n)
            else:
                Com.cg_grad(gtemp, xtemp, n)
                Com.f = Com.cg_value(xtemp, n)
            Com.df = cg_dot(gtemp, d, n)
            Com.nf += 1
            Com.ng += 1

            # Reduce step size if function or derivative is NaN
            if (not np.isnan(Com.df)) or (not np.isnan(Com.f)):
                for i in range(Parm.nexpand):
                    alpha *= Parm.nan_decay
                    cg_step(xtemp, x, d, alpha, n)
                    if Com.cg_valgrad != None:
                        Com.f = Com.cg_valgrad(gtemp, xtemp, n)
                    else:
                        Com.cg_grad(gtemp, xtemp, n)
                        Com.f = Com.cg_value(xtemp, n)
                    Com.df = cg_dot(gtemp, d, n)
                    Com.nf += 1
                    Com.ng += 1
                    if np.isnan(Com.df) and np.isnan(Com.f):
                        break
                if i == Parm.nexpand:
                    return -2
                Com.rho = Parm.nan_rho
            else:
                Com.rho = Parm.rho
            Com.alpha = alpha
            
    else:  # cg_evaluate without NaN checking
        if what == "fg":  # Compute both function and gradient
            if alpha == 0.0:  # cg_evaluate at x
                if Com.cg_valgrad != None:
                    Com.f = Com.cg_valgrad(Com.g, x, n)
                else:
                    Com.cg_grad(Com.g, x, n)
                    Com.f = Com.cg_value(x, n)
            else:
                cg_step(xtemp, x, d, alpha, n)
                if Com.cg_valgrad != None:
                    Com.f = Com.cg_valgrad(gtemp, xtemp, n)
                else:
                    Com.cg_grad(gtemp, xtemp, n)
                    Com.f = Com.cg_value(xtemp, n)
                Com.df = cg_dot(gtemp, d, n)
            Com.nf += 1
            Com.ng += 1
        elif what == "f":  # Compute function only
            cg_step(xtemp, x, d, alpha, n)
            Com.f = Com.cg_value(xtemp, n)
            Com.nf += 1
        else:  # Compute gradient only
            cg_step(xtemp, x, d, alpha, n)
            Com.cg_grad(gtemp, xtemp, n)
            Com.df = cg_dot(gtemp, d, n)
            Com.ng += 1

    return 0

########################################################################

# Com = ls.CGCom

def cg_contract(A, fA, dA, B, fB, dB, Com):
    # Parm = lsu.CGParameter()
    AWolfe = Com.AWolfe
    Parm = Com.Parm
    PrintLevel = Parm.PrintLevel
    a = A
    fa = fA
    da = dA
    b = B
    fb = fB
    db = dB
    f1 = fb
    d1 = db
    toggle = 0
    width = 0.0
    
    for iter in range(Parm.nshrink):
        if toggle == 0 or (toggle == 2 and (b - a) <= width):
            alpha = cg_cubic(a, fa, da, b, fb, db)
            toggle = 0
            width = Parm.gamma * (b - a)
            if iter:
                Com.QuadOK = True
        elif toggle == 1:
            Com.QuadOK = True
            if old < a:
                alpha = cg_cubic(a, fa, da, old, fold, dold)
            else:
                alpha = cg_cubic(a, fa, da, b, fb, db)
        else:
            alpha = 0.5 * (a + b)
            Com.QuadOK = False

        if alpha <= a or alpha >= b:
            alpha = 0.5 * (a + b)
            Com.QuadOK = False

        toggle += 1
        if toggle > 2:
            toggle = 0

        Com.alpha = alpha
        cg_evaluate("fg", "n", Com)
        f = Com.f
        df = Com.df

        if Com.QuadOK:
            if cg_Wolfe(alpha, f, df, Com):
                return 0
        if not AWolfe:
            f -= alpha * Com.wolfe_hi
            df -= Com.wolfe_hi

        if df >= 0.0:
            B = alpha
            fB = f
            dB = df
            A = a
            fA = fa
            dA = da
            return -2

        if f <= Com.fpert:
            old = a
            a = alpha
            fold = fa
            fa = f
            dold = da
            da = df
        else:
            old = b
            b = alpha
            fb = f
            db = df

        if PrintLevel >= 2:
            s = "OK" if Com.QuadOK else ""
            print(f"contract  {s:>2} a: {a:13.6e} b: {b:13.6e} fa: {fa:13.6e} fb: {fb:13.6e} da: {da:13.6e} db: {db:13.6e}")

    if abs(fb) <= Com.SmallCost:
        Com.PertRule = False

    t = Com.f0
    if Com.PertRule:
        if t != 0.0:
            Com.eps = Parm.egrow * (f1 - t) / abs(t)
            Com.fpert = t + abs(t) * Com.eps
        else:
            Com.fpert = 2.0 * f1
    else:
        Com.eps = Parm.egrow * (f1 - t)
        Com.fpert = t + Com.eps

    if PrintLevel >= 1:
        print(f"--increase eps: {Com.eps:e} fpert: {Com.fpert:e}")

    Com.neps += 1
    return -1


########################################################################

# Com = ls.CGCom

def cg_line(Com):
    # Parm = lsu.CGParameter()
    AWolfe = Com.AWolfe
    Parm = Com.Parm
    PrintLevel = Parm.PrintLevel
    Line = False

    if PrintLevel >= 1:
        if AWolfe:
            print("Approximate Wolfe line search")
            print("=============================")
        else:
            print("Wolfe line search")
            print("=================")
        
    b = Com.alpha
    if Com.QuadOK:
        status = cg_evaluate("fg", "y", Com)
        fb = Com.f 
        if not AWolfe:
            fb -= b * Com.wolfe_hi
        qb = True
    else:
        status = cg_evaluate("g", "y", Com)
        qb = False

    if status:
        return status  # return if function is NaN
    
    if AWolfe:
        db = Com.df
        d0 = da = Com.df0
    else: 
        db = Com.df - Com.wolfe_hi
        d0 = da = Com.df0 - Com.wolfe_hi

    a = 0.0
    a1 = 0.0
    d1 = d0
    fa = Com.f0
    
    if PrintLevel >= 1:
            fmt1 = "{:9s} {:2s} a: {:13.6e} b: {:13.6e} fa: {:13.6e} fb: {:13.6e} da: {:13.6e} db: {:13.6e}"
            fmt2 = "{:9s} {:2s} a: {:13.6e} b: {:13.6e} fa: {:13.6e} fb: x.xxxxxxxxxx da: {:13.6e} db: {:13.6e}"
            s2 = "OK" if Com.QuadOK else ""
                
            if qb:
                print(fmt1.format("start", s2, a, b, fa, fb, da, db))
            else:
                print(fmt2.format("start", s2, a, b, fa, da, db))
                
    if Com.QuadOK and Com.f <= Com.f0:
        if cg_Wolfe(b, Com.f, Com.df, Com):
            return 0

    if not AWolfe:
        Com.Wolfe = True

    rho = Com.rho
    ngrow = 1

    while db < 0.0:
        if not qb:
            cg_evaluate("f", "n", Com)
            if AWolfe:
                fb = Com.f  
            else:
                fb = Com.f - b * Com.wolfe_hi
            qb = True

        if fb > Com.fpert:
            status = cg_contract(a, fa, da, b, fb, db, Com)
            if status == 0:
                return 0
            if status == -2:
                Line = True
                break
            if Com.neps > Parm.neps:
                return 6

        ngrow += 1
        if ngrow > Parm.nexpand:
            return 3

        a, fa, da = b, fb, db
        d2, d1 = d1, da
        a2, a1 = a1, a

        if ngrow in {3, 6}:
            if d1 > d2:
                secant = d1 / (d1 - d2)
                
                if (d1 - d2) / (a1 - a2) >= (d2 - d0) / a2:
                    b = a1 - (a1 - a2) * secant
                else:
                    b = a1 - Parm.SecantAmp * (a1 - a2) * secant
                b = min(b, Parm.ExpandSafe * a1)
            else:
                rho *= Parm.RhoGrow
                b = rho * b
        else:
            b = rho * b
        Com.alpha = b
        cg_evaluate("g", "n", Com)
        qb = False
        db = Com.df if AWolfe else Com.df - Com.wolfe_hi

        if PrintLevel >= 2:
            s2 = "OK" if Com.QuadOK else ""
            print(fmt2.format("expand   ", s2, a, b, fa, da, db))
   
    # if Line:
    toggle = 0
    width = b - a
    qb0 = False

    for iter in range(Parm.nline):
        if toggle == 0 or (toggle == 2 and (b - a) <= width):
            Com.QuadOK = True
            if Com.UseCubic and qb:
                s1 = "cubic    "
                alpha = cg_cubic(a, fa, da, b, fb, db)
                if alpha < 0.0:
                    s1 = "secant   " 
                    if -da < db:
                        alpha = a - (a-b)*(da/(da-db)) 
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db)) 
                    else:
                        alpha = -1.0
            else:
                s1 = "secant   " 
                if -da < db:
                    alpha = a - (a-b)*(da/(da-db)) 
                elif da != db:
                    alpha = b - (a-b)*(db/(da-db)) 
                else:
                    alpha = -1.0
            width = Parm.gamma * (b - a)
            
        elif toggle == 1:
            Com.QuadOK = True
            if Com.UseCubic:
                s1 = "cubic    "
                if Com.alpha == a:
                    alpha = cg_cubic(a0, fa0, da0, a, fa, da)
                elif qb0:
                    alpha = cg_cubic (b, fb, db, b0, fb0, db0) 
                else:
                    alpha = -1.0
                    
                if alpha <= a or alpha >= b:
                    if qb:
                        alpha = cg_cubic(a, fa, da, b, fb, db)  
                    else:
                        alpha = -1
                
                if alpha < 0.0:
                    s1 = "secant   "
                    if -da < db:
                        alpha = a - (a-b)*(da/(da-db)) 
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db)) 
                    else:
                        alpha = -1.0  
                        
            else:
                s1 = "secant   " 
                if (Com.alpha == a) and da > da0:
                    alpha = a - (a-a0)*(da/(da-da0)) 
                elif db < db0:
                    alpha = b - (b-b0)*(db/(db-db0))
                else:
                    if -da < db:
                        alpha = a - (a-b)*(da/(da-db))
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db))
                    else:
                        alpha = -1.0

                if alpha <= a or alpha >= b:
                    if -da < db:
                        alpha = a - (a-b)*(da/(da-db))
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db)) 
                    else:
                        alpha = -1.0
        
        else:   
            alpha = 0.5 * (a + b)
            s1 = "bisection"
            Com.QuadOK = False

        if alpha <= a or alpha >= b:
            alpha = 0.5 * (a + b)
            s1 = "bisection"
            if alpha == a or alpha == b:
                return 7
            Com.QuadOK = False

        if toggle == 0:
            a0, b0 = a, b
            da0, db0 = da, db
            fa0 = fa
            if qb:
                fb0 = fb
                qb0 = True

        toggle += 1
        if toggle > 2:
            toggle = 0

        Com.alpha = alpha
        cg_evaluate("fg", "n", Com)
        Com.alpha = alpha
        f = Com.f
        df = Com.df
        
        if Com.QuadOK:
            if cg_Wolfe(alpha, f, df, Com):
                if PrintLevel >= 2:
                    print(f"             a: {alpha:13.6e} f: {f:13.6e} df: {df:13.6e} {s1:>1}")
                return 0

        if not AWolfe:
            f -= alpha * Com.wolfe_hi
            df -= Com.wolfe_hi

        if df >= 0.0:
            b, fb, db = alpha, f, df
            qb = True
        elif f <= Com.fpert:
            a, da, fa = alpha, df, f
        else:
            B = b
            if qb:
                fB = fb
            dB = db
            b, fb, db = alpha, f, df
            status = cg_contract(a, fa, da, b, fb, db, Com)
            if status == 0:
                return 0
            if status == -1:
                if Com.neps > Parm.neps:
                    return 6
                a, fa, da = b, fb, db
                b = B
                if qb:
                    fb = fB
                db = dB
            else:
                qb = True

        if PrintLevel >= 2:
            s2 = "OK" if Com.QuadOK else "" 
            if not qb:
                print(fmt2.format(s1, s2, a, b, fa, da, db)) 
            else:
                print(fmt1.format(s1, s2, a, b, fa, fb, da, db))
    return 4

##########################################################################

# Com = ls.CGCom

def cg_Wolfe(alpha, f, dphi, Com):
    if dphi >= Com.wolfe_lo:
        
        if f - Com.f0 <= alpha * Com.wolfe_hi:
            if Com.Parm.PrintLevel >= 2:
                print("Wolfe conditions hold")
            return 1
        
        elif Com.AWolfe:
            if f <= Com.fpert and dphi <= Com.awolfe_hi:
                if Com.Parm.PrintLevel >= 2:
                    print("Approximate Wolfe conditions hold")
                return 1  
    
    return 0  


######################################################################### 

# Stat, UParm = lsu.CGStats, lsu.CGParameter

def line_search(x, n, dir, Stat, UParm, value, grad, valgrad):
    
    while True:    
        Parm = lsu.CGParameter
        ParmStruct = lsu.CGParameter()
        Com = ls.CGCom()
        
        exit = False

        if UParm is None:
            Parm = ParmStruct
            cg_default(Parm)
        else:
            Parm = UParm
            
        PrintLevel = Parm.PrintLevel
        Com.Parm = Parm
        Com.eps = Parm.eps
        Com.PertRule = Parm.PertRule
        Com.Wolfe = False
        QuadF = False

        if Parm.PrintParms:
            cg_printParms(Parm)

        work = np.zeros(4 * n)
        if work is None:
            print(f"Insufficient memory for specified problem dimension {n}")
            status = 9
            return status
        


        Com.x = x
        Com.d = d = work
        Com.g = g = d + n
        Com.xtemp = xtemp = g + n
        Com.gtemp = gtemp = xtemp + n
        Com.n = n
        Com.nf = 0
        Com.ng = 0
        Com.neps = 0
        Com.AWolfe = Parm.AWolfe
        Com.cg_value = value
        Com.cg_grad = grad
        Com.cg_valgrad = valgrad
        StopRule = Parm.StopRule

    
        f = 0.0
        fbest = INF
        gbest = INF
        nslow = 0
        slowlimit = 2 * n + Parm.nslow
        n5 = n % 5
        
        Ck = 0.0
        Qk = 0.0

        Com.alpha = 0.0
        cg_evaluate("fg", "n", Com)
        f = Com.f
        Com.f0 = f + f
        Com.SmallCost = abs(f) * Parm.SmallCost
        xnorm = 0.0
        for i in range(n5):
            if xnorm < abs(x[i]):
                xnorm = abs(x[i])
        for i in range(n5, n, 5):
            if xnorm < abs (x [i]  ): xnorm = abs (x [i]  ) 
            if xnorm < abs (x [i+1]): xnorm = abs (x [i+1]) 
            if xnorm < abs (x [i+2]): xnorm = abs (x [i+2]) 
            if xnorm < abs (x [i+3]): xnorm = abs (x [i+3]) 
            if xnorm < abs (x [i+4]): xnorm = abs (x [i+4])
            # xnorm = max(xnorm, abs(x[i]), abs(x[i+1]), abs(x[i+2]),
            #             abs(x[i+3]), abs(x[i+4]))

        gnorm = 0.0
        gnorm2 = 0.0
        dnorm2 = 0.0
        
        for i in range(n5):
            t = g[i]
            gnorm2 += t * t
            if gnorm < abs (t): gnorm = abs (t)
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]
        
        i = n5
        while  i < n:
        # for i in range (n5, n):
            t = g[i]
            gnorm2 += t * t
            if gnorm < abs (t): gnorm = abs (t)
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]
            i += 1
            
            t = g[i]
            gnorm2 += t * t
            if gnorm < abs (t): gnorm = abs (t)
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]
            i += 1
            
            t = g[i]
            gnorm2 += t * t
            if gnorm < abs (t): gnorm = abs (t)
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]
            i += 1
            
            t = g[i]
            gnorm2 += t * t
            if gnorm < abs (t): gnorm = abs (t)
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]
            i += 1
            
            t = g[i]
            gnorm2 += t * t
            if gnorm < abs (t): gnorm = abs (t)
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]
            i += 1
            
        
            
        if not np.isnan(f):
            status = -1
            exit = True
            break

        if PrintLevel >= 1:
            print(f"first: {0:5d} f: {f:13.6e} df: {-gnorm2:13.6e} gnorm: {gnorm:13.6e}")
        
        if cg_tol(gnorm, Com):
            status = 0
            exit = True
            break
        
        dphi0 = cg_dot(g, d, n)
        if dphi0 > 0.0:
            status = 5
            exit = True
            break
        
        delta2 = 2 * Parm.delta - 1.0
        alpha = Parm.step
        if alpha == 0.0:
            alpha = Parm.psi0 * xnorm / gnorm
            if xnorm == 0.0:
                if f != 0.0:
                    alpha = Parm.psi0 * abs(f) / gnorm2
                else:
                    alpha = 1.0
        
        IterRestart = 0
        IterQuad = 0
        
        Com.QuadOK = False
        alpha = Parm.psi2 * alpha
        
        if f != 0.0:
            t = abs((f - Com.f0) / f)
        else:
            t = 1.0
                    
        Com.UseCubic = True
        
        if t < Parm.CubicCutOff or (not Parm.UseCubic):
            Com.UseCubic = False
        if Parm.QuadStep:
            if (t > Parm.QuadCutOff and abs(f) >= Com.SmallCost) or QuadF:
                
                Com.palha = Parm.psi1 * alpha
                if QuadF:
                    status = cg_evaluate("g", "y", Com)
                    if status:
                        exit = True
                        break
                    if Com.df > dphi0:
                        alpha = -dphi0 / ((Com.df - dphi0) / Com.alpha)
                        Com.QuadOK = True
                else:
                    status = cg_evaluate("f", "y", Com)
                    if status:
                        exit = True
                        break
                    ftemp = Com.f
                    denom = 2.0 * (((ftemp - f) / Com.alpha) - dphi0)
                    if denom > 0.0:
                        t = -dphi0 * Com.alpha / denom
                        alpha = t 
                        if ftemp < f or QuadF:
                            Com.QuadOK = True
                        else:
                            alpha = max(t, Com.alpha * Parm.QuadSafe)
                if PrintLevel >= 1:
                    if denom <= 0.0:
                        print(f"Quad step fails (denom = {denom:14.6e})")
                    elif Com.QuadOK:
                        print(f"Quad step {alpha:14.6e} OK")
                    else:
                        print(f"Quad step {alpha:14.6e} done, but not OK") 
        
            elif PrintLevel >= 1:
                print(f"No quad step (chg: {t:14.6e}, cut: {Parm.QuadCutOff:10.2e})")
        
        
        Com.f0 = f 
        Com.df0 = dphi0
        

        Qk = Parm.Qdecay * Qk + 1.0
        Ck = Ck + (abs(f) - Ck) / Qk
    
        if Com.PertRule:
            Com.fpert = f + Com.eps * abs(f)
        else:
            Com.fpert = f + Com.eps 

        Com.wolfe_hi = Parm.delta * dphi0
        Com.wolfe_lo = Parm.sigma * dphi0
        Com.awolfe_hi = delta2 * dphi0 
        Com.alpha = alpha

        status = cg_line(Com)

        if status > 0 and (not Com.AWolfe):
            if PrintLevel >= 1:
                print(f"\nWOLFE LINE SEARCH FAILS")
            if status != 3:
                Com.AWolfe = True
                status = cg_line(Com)

        alpha = Com.alpha 
        f = Com.f 
        dphi = Com.df

        cg_copy(x,xtemp,n)

        if status:
            exit = True
            break

        if (not np.isnan(f)) or (not np.isnan(dphi)):
            status = 10 
            exit = True
            break

        if -alpha * dphi0 <= Parm.feps * abs(f):
            status = 1
            exit = True
            break
        
        break
        
        
    # if exit:   
    # Stat = lsu.CGStats
    # Com = ls.CGCom     
    if Stat != None:
        Stat.f = f
        Stat.gnorm = gnorm 
        Stat.nfunc = Com.nf
        Stat.ngrad = Com.ng
        for i in range(n):
            Stat.g[i] = gtemp[i]
        Stat.alpha = Com.alpha

    if status > 2:
        gnorm = 0.0
        for i in range(n):
            x[i] = xtemp[i]
            g[i] = gtemp[i]
            t = abs(g[i])
            gnorm = max(gnorm, t)
        if Stat != None:
            Stat.gnorm = gnorm
    if Parm.PrintFinal or PrintLevel >= 1:
        mess1 = "Possible causes of this error message:" 
        mess2 = "   - your tolerance may be too strict: grad_tol = "
        mess3 = "Line search fails"
        mess4 = "   - your gradient routine has an error"
        mess5 = "   - the parameter epsilon in cg_descent_c.Parm is too small"
        print(f"\nTermination status: {status}")
        if status == -2:
            print("function value became nan") 

        elif status == -1:
            print("Objective function value is nan at starting point")
        

        elif status == 0:
            print("Convergence")

        elif status == 1:
            print("Terminating since change in function value <= feps*|f|")

        elif status == 2:
            pass
            # print("Number of iterations exceed specified limit")
            # print(f"Iterations: {iter:10.0f} maxit: {maxit:10.0f}"
            # print(f"{messs1:s}")
            # print(f"{medd2:s} {grad_tol:e}")

        elif status == 3:
            print("Slope always negative in line search")
            print(f"{mess1}")
            print("   - your cost function has an error")
            print(f"{mess4}")

        elif status == 4:
            print("Line search fails, too many iterations")

        elif status == 5:
            print("Search direction not a descent direction")

        elif status == 6:
            print(f"{mess3}") 
            print(f"{mess1}")
            print(f"{mess4}")
            print(f"{mess5}")
            
        elif status == 7:
            print(f"{mess3}")
            print(f"{mess1}")

        elif status == 8:
            print(f"{mess3}")
            print(f"{mess1}")
            print(f"{mess4}")
            print(f"{mess5}")
        
        elif status == 8:
            print("Debugger is on, function value does not improve")
            print(f"new value: {f:25.16e} old value: {Com.f0:25.16e}")
        
        elif status == 9:
            print("Insufficient memory")
        
        elif status == 10:
            print("Function beCome nan")

        elif status == 11:
            print("{nslow} iterations without strict improvement in cost or gradient\n")

        # print(f"maximum norm for gradient: {gnorm:13.6e}")
        # print(f"function value:            {f:13.6e}\n")
        # print(f"cg  iterations:          {iter:10.0f}")
        # print(f"function evaluations:    {Com.nf:10.0f}")
        # print(f"gradient evaluations:    {Com.ng:10.0f}")
        # print("===================================\n")

    return status
        
