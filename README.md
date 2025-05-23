# Wolfe Line Search - جستجوی خطی با شرایط Wolfe

این پروژه یک پیاده‌سازی کامل از الگوریتم جستجوی خطی با شرایط Wolfe است که در بهینه‌سازی عددی و روش‌های گرادیان مزدوج (Conjugate Gradient) کاربرد دارد.

## ساختار پروژه

پروژه شامل چهار فایل اصلی است:

1. **line_search_algorithm.py**  

2. **line_search_user.py**  

3. **line_search.py**  
   
4. **gradient_method.py**  

---

## 1. line_search_algorithm.py

### توضیح کلی  
این فایل شامل توابع اصلی و پایه‌ای برای پیاده‌سازی الگوریتم جستجوی خطی Wolfe است. در این فایل، توابعی برای تنظیم پارامترهای الگوریتم، ضرب داخلی بردارها، کپی‌برداری، گام برداشتن، ارزیابی تابع هدف و گرادیان، و همچنین اجرای مراحل مختلف جستجوی خطی پیاده‌سازی شده‌اند. این توابع به عنوان هسته محاسباتی الگوریتم عمل می‌کنند.

### کد کامل فایل

````python
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
            fmt1 = "{:>9} {:>2} a: {:13.6e} b: {:13.6e} fa: {:13.6e} fb: {:13.6e} da: {:13.6e} db: {:13.6e}"
            fmt2 = "{:>9} {:>2} a: {:13.6e} b: {:13.6e} fa: {:13.6e} fb: x.xxxxxxxxxx da: {:13.6e} db: {:13.6e}"
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
   
    if Line:
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
        # Parm = lsu.CGParameter
        ParmStruct = lsu.CGParameter()
        Com = ls.CGCom
        
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
            # gnorm = max(gnorm, abs(t))
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]

        for i in range (n5, n, 5):
            for j in range (5):
                t = g[i]
                gnorm2 += t * t
                if gnorm < abs (t): gnorm = abs (t)
                # gnorm = max(gnorm, abs(t))
                d[i] = dir[i]
                dnorm2 += d[i] * d[i]
            
        
            
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
        
        
    if exit:   
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
        


````
### توضیحات بخشی به بخش کد

````python
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


````
این تابع (cg_default) تمام پارامترهای پیش‌فرض مورد نیاز برای الگوریتم گرادیان مزدوج (Conjugate Gradient) و جستجوی خطی Wolfe را مقداردهی اولیه می‌کند.
هر خط یک ویژگی از شیء Parm را مقداردهی می‌کند که این ویژگی‌ها رفتار الگوریتم را کنترل می‌کنند.
توضیح برخی پارامترهای مهم:

PrintFinal, PrintLevel, PrintParms: کنترل چاپ خروجی و سطح جزئیات.
AWolfe, AWolfeFac: فعال‌سازی و تنظیم جستجوی Wolfe تقریبی.
Qdecay, nslow: کنترل نرخ کاهش و تعداد تکرارهای بدون پیشرفت.
StopRule, StopFac: قوانین توقف الگوریتم.
PertRule, eps, egrow: تنظیمات مربوط به دقت و خطا.
QuadStep, QuadCutOff, QuadSafe: فعال‌سازی و کنترل گام‌های درجه دوم (quadratic).
UseCubic, CubicCutOff: فعال‌سازی و کنترل گام‌های درجه سوم (cubic).
SmallCost: آستانه برای هزینه‌های کوچک.
step, psi0, psi1, psi2: پارامترهای اولیه برای گام و تخمین اولیه.
AdaptiveBeta, BetaLower, theta: تنظیمات مربوط به به‌روزرسانی پارامتر بتا در الگوریتم.
maxit_fac, neps, nshrink, nline, nexpand: محدودیت‌های تکرار و تعداد دفعات مجاز برای عملیات مختلف.
delta, sigma, gamma, rho, nan_rho, nan_decay: پارامترهای کنترل جستجوی خطی و مدیریت شرایط خاص (مانند NaN).
qeps, qrestart, qrule: پارامترهای مربوط به تشخیص رفتار درجه دوم تابع هدف.
این تابع معمولاً در ابتدای اجرای الگوریتم فراخوانی می‌شود تا همه پارامترها مقدار مناسبی داشته باشند.

````python
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


````
تابع cg_printParms(Parm) برای چاپ تمام پارامترهای ساختار Parm به صورت خوانا و دسته‌بندی‌شده استفاده می‌شود. این تابع، مقادیر فعلی پارامترهای الگوریتم جستجوی خطی Wolfe و گرادیان مزدوج را نمایش می‌دهد تا کاربر بتواند تنظیمات فعلی را مشاهده و بررسی کند.

عملکرد بخش‌های مختلف تابع:

ابتدا پارامترهای عددی مهم مانند delta، sigma، gamma، rho و ... را با توضیح کوتاه چاپ می‌کند.
سپس پارامترهای منطقی (Boolean) را با توضیح عملکرد هرکدام نمایش می‌دهد (مثلاً آیا از گام درجه دوم یا سوم استفاده می‌شود یا خیر).
در صورت فعال بودن برخی ویژگی‌ها (مانند Wolfe تقریبی)، پیام مناسب چاپ می‌شود.
در انتها، وضعیت پارامترهای توقف و اشکال‌زدایی نیز نمایش داده می‌شود.
این تابع برای دیباگ و اطمینان از صحت مقداردهی پارامترهای الگوریتم بسیار مفید است و معمولاً قبل از شروع حل یا هنگام بررسی رفتار الگوریتم فراخوانی می‌شود.

````python
def cg_dot(x, y, n):
    t = 0.0
    n5 = n % 5
    for i in range(n5):
        t += x[i] * y[i]
    for i in range(n5, n, 5):
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2] + x [i+3]*y [i+3] + x [i+4]*y [i+4]
        # t += np.dot(x[i:i+5], y[i:i+5])
    return t

````
تابع cg_dot(x, y, n) ضرب داخلی (dot product) دو بردار را با طول n محاسبه می‌کند، اما برای افزایش سرعت، جمع را به صورت بلوکی (در دسته‌های ۵تایی) انجام می‌دهد.

توضیح عملکرد:

ابتدا باقیمانده n تقسیم بر ۵ (n5 = n % 5) را حساب می‌کند تا عناصر ابتدایی که در یک بلوک ۵تایی جا نمی‌گیرند، جداگانه جمع شوند.
حلقه اول: از ۰ تا n5، ضرب داخلی عناصر ابتدایی را محاسبه می‌کند.
حلقه دوم: از n5 تا n، با گام ۵، هر بار ۵ عنصر را با هم جمع می‌کند تا سرعت بیشتر شود.
در نهایت مقدار کل ضرب داخلی را برمی‌گرداند.
این روش نسبت به حلقه ساده، برای آرایه‌های بزرگ سریع‌تر است و مشابه تکنیک‌های بهینه‌سازی حلقه (loop unrolling) در زبان‌های سطح پایین است.
خروجی این تابع معادل با np.dot(x, y) است.

````python
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

````
تابع cg_copy(y, x, n) برای کپی کردن عناصر آرایه x به آرایه y با طول n استفاده می‌شود.
این تابع برای افزایش سرعت، کپی را به صورت بلوکی (در دسته‌های ۱۰تایی) انجام می‌دهد.

توضیح عملکرد:

ابتدا باقیمانده n تقسیم بر ۱۰ (n10 = n % 10) را حساب می‌کند تا عناصر ابتدایی که در یک بلوک ۱۰تایی جا نمی‌گیرند، جداگانه کپی شوند.
حلقه اول: از ۰ تا n10، عناصر ابتدایی را کپی می‌کند.
حلقه دوم: از n10 تا n، با گام ۱۰، هر بار ۱۰ عنصر را با هم کپی می‌کند تا سرعت بیشتر شود.
این روش مشابه تکنیک‌های بهینه‌سازی حلقه (loop unrolling) است و برای آرایه‌های بزرگ سریع‌تر از حلقه ساده عمل می‌کند.
خروجی این تابع معادل با np.copyto(y[:n], x[:n]) است.

````python
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


````
تابع cg_step(xtemp, x, d, alpha, n) برای محاسبه گام بعدی در الگوریتم بهینه‌سازی استفاده می‌شود. این تابع مقدار هر عنصر از بردار جدید xtemp را به صورت x[i] + alpha * d[i] محاسبه می‌کند.

توضیح عملکرد:

ابتدا باقیمانده n تقسیم بر ۵ (n5 = n % 5) را حساب می‌کند تا عناصر ابتدایی که در یک بلوک ۵تایی جا نمی‌گیرند، جداگانه محاسبه شوند.
حلقه اول: از ۰ تا n5، مقدار هر عنصر را محاسبه و در xtemp قرار می‌دهد.
حلقه دوم: از n5 تا n، با گام ۵، هر بار ۵ عنصر را با هم محاسبه می‌کند تا سرعت بیشتر شود.
این روش مشابه تکنیک‌های بهینه‌سازی حلقه (loop unrolling) است و برای آرایه‌های بزرگ سریع‌تر از حلقه ساده عمل می‌کند.
خروجی این تابع معادل با xtemp[:] = x + alpha * d است.

````python
def cg_tol(gnorm, Com):
    if Com.Parm.StopRule:
        if gnorm <= Com.tol:
            return 1
    elif gnorm <= Com.tol * (1.0 + abs(Com.f)): return 1
    return 0

````
تابع cg_tol(gnorm, Com) بررسی می‌کند که آیا شرط توقف الگوریتم بر اساس نُرم گرادیان (gnorm) برقرار است یا نه.

توضیح عملکرد:

اگر پارامتر StopRule در ساختار پارامترها (Com.Parm.StopRule) فعال باشد:
اگر مقدار نُرم گرادیان (gnorm) از مقدار آستانه تحمل (Com.tol) کمتر یا مساوی باشد، مقدار 1 (یعنی توقف) برمی‌گرداند.
اگر StopRule فعال نباشد:
اگر نُرم گرادیان از Com.tol * (1.0 + abs(Com.f)) کمتر یا مساوی باشد، مقدار 1 برمی‌گرداند.
در غیر این صورت، مقدار 0 (یعنی ادامه الگوریتم) برمی‌گرداند.
این تابع برای کنترل همگرایی و توقف به موقع الگوریتم بهینه‌سازی استفاده می‌شود. 

````python
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


````
تابع cg_cubic(a, fa, da, b, fb, db) برای تقریب ریشه (یا نقطه بهینه) یک چندجمله‌ای درجه سه (cubic interpolation) بین دو نقطه a و b با مقادیر تابع (fa, fb) و مشتق‌ها (da, db) استفاده می‌شود. این تابع معمولاً در جستجوی خطی (line search) برای تخمین مقدار بهینه گام (step size) کاربرد دارد.

توضیح گام به گام:

اختلاف بین نقاط را با delta = b - a محاسبه می‌کند. اگر صفر باشد، مقدار a را بازمی‌گرداند.
مقدار v را بر اساس مشتق‌ها و اختلاف مقادیر تابع محاسبه می‌کند.
مقدار t را به عنوان معیاری برای تعیین ریشه‌های حقیقی یا مختلط محاسبه می‌کند.
اگر t < 0.0 باشد (ریشه مختلط)، روش سکانت (secant) را برای تقریب استفاده می‌کند.
اگر ریشه حقیقی باشد، مقدار w را با جذر t محاسبه می‌کند (با توجه به جهت delta).
دو مقدار کمکی d1 و d2 را محاسبه می‌کند.
اگر هر دو صفر باشند، مقدار -1.0 بازمی‌گرداند (خطا).
در نهایت، بسته به بزرگ‌تر بودن قدرمطلق d1 یا d2، نقطه بهینه را با یکی از دو فرمول cubic بازمی‌گرداند.
این تابع باعث می‌شود تخمین گام در جستجوی خطی دقیق‌تر و سریع‌تر به نقطه بهینه همگرا شود.

````python
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

````
تابع cg_evaluate(what, nan, Com) وظیفه ارزیابی مقدار تابع هدف و/یا گرادیان را در نقطه فعلی یا نقطه جدید (بر اساس گام و جهت حرکت) بر عهده دارد و در صورت نیاز، مدیریت شرایط NaN یا Inf را انجام می‌دهد.

توضیح عملکرد:

ورودی‌ها:

what: مشخص می‌کند چه چیزی باید محاسبه شود ("f" فقط مقدار تابع، "g" فقط گرادیان، "fg" هر دو).
nan: اگر "y" باشد، بررسی NaN/Inf انجام می‌شود و در صورت نیاز گام کاهش می‌یابد.
Com: ساختار داده شامل متغیرها، جهت حرکت، پارامترها و توابع هدف/گرادیان.
اگر بررسی NaN فعال باشد:

مقدار تابع یا گرادیان (یا هر دو) را در نقطه جدید محاسبه می‌کند.
اگر مقدار به دست آمده NaN یا Inf باشد، گام (alpha) را با ضریب nan_decay کوچک می‌کند و دوباره ارزیابی می‌کند (تا حداکثر nexpand بار).
اگر پس از این تلاش‌ها هنوز مقدار معتبر به دست نیاید، مقدار خطا (-2) بازمی‌گرداند.
مقدار جدید alpha و در صورت نیاز rho به‌روزرسانی می‌شود.
اگر بررسی NaN فعال نباشد:

بر اساس مقدار what، فقط مقدار تابع، فقط گرادیان یا هر دو را محاسبه می‌کند.
شمارنده‌های تعداد ارزیابی تابع (nf) و گرادیان (ng) به‌روزرسانی می‌شوند.
در نهایت، اگر همه چیز درست باشد، مقدار ۰ بازگردانده می‌شود.

این تابع برای اطمینان از معتبر بودن مقدار تابع هدف و گرادیان در طول جستجوی خطی و جلوگیری از توقف ناگهانی الگوریتم بهینه‌سازی به دلیل NaN/Inf استفاده می‌شود.

````python
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

````
تابع cg_contract یک بخش کلیدی از الگوریتم جستجوی خطی Wolfe است که وظیفه "انقباض بازه" (contracting the interval) را در هنگام جستجوی مقدار مناسب گام (step size) بر عهده دارد.

توضیح عملکرد:

مقداردهی اولیه:
متغیرهای محلی برای بازه [a, b] و مقادیر تابع و مشتق در ابتدا و انتهای بازه تنظیم می‌شوند.

حلقه انقباض:
تا حداکثر Parm.nshrink بار، بازه [a, b] را کوچک می‌کند تا شرایط Wolfe برقرار شود یا بازه به اندازه کافی کوچک شود.

بسته به مقدار toggle، گام جدید (alpha) با روش cubic، یا secant، یا میانه بازه انتخاب می‌شود.
اگر مقدار جدید alpha خارج از بازه باشد، مقدار میانه بازه انتخاب می‌شود.
مقدار toggle برای تنوع روش انتخاب گام تغییر می‌کند.
ارزیابی تابع و گرادیان:
مقدار تابع و مشتق در نقطه جدید محاسبه می‌شود.

بررسی شرایط Wolfe:
اگر شرایط Wolfe برقرار باشد، تابع با مقدار 0 خاتمه می‌یابد (موفقیت).

به‌روزرسانی بازه:
اگر مشتق مثبت شود (یعنی دیگر جهت نزولی نیست)، بازه و مقادیر به‌روزرسانی و تابع با مقدار -2 خاتمه می‌یابد. اگر مقدار تابع از آستانه مجاز کمتر باشد، بازه از سمت چپ کوچک می‌شود؛ در غیر این صورت از سمت راست.

در پایان حلقه:
اگر مقدار تابع در انتهای بازه خیلی کوچک باشد، قاعده خطا (PertRule) غیرفعال می‌شود. سپس مقدار eps و fpert برای کنترل خطا به‌روزرسانی می‌شوند و شمارنده افزایش می‌یابد.

خروج:
اگر هیچ‌کدام از شرایط موفقیت برقرار نشود، مقدار -1 بازگردانده می‌شود (یعنی نیاز به افزایش تحمل خطا یا ادامه جستجو).

این تابع برای اطمینان از همگرایی و کنترل دقیق گام در جستجوی خطی استفاده می‌شود و نقش مهمی در پایداری و دقت الگوریتم دارد.

````python
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
            fmt1 = "{:>9} {:>2} a: {:13.6e} b: {:13.6e} fa: {:13.6e} fb: {:13.6e} da: {:13.6e} db: {:13.6e}"
            fmt2 = "{:>9} {:>2} a: {:13.6e} b: {:13.6e} fa: {:13.6e} fb: x.xxxxxxxxxx da: {:13.6e} db: {:13.6e}"
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
   
    if Line:
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

````
تابع cg_line(Com) هسته اصلی جستجوی خطی Wolfe است و مقدار بهینه گام (alpha) را با رعایت شرایط Wolfe یا Wolfe تقریبی پیدا می‌کند.

خلاصه عملکرد:

آغاز و مقداردهی اولیه:

پارامترها و متغیرهای محلی مقداردهی می‌شوند.
اگر سطح چاپ فعال باشد، نوع جستجو (Wolfe یا تقریبی) چاپ می‌شود.
ارزیابی اولیه:

اگر گام درجه دوم (QuadOK) فعال باشد، مقدار تابع و گرادیان در نقطه اولیه محاسبه می‌شود.
در غیر این صورت فقط گرادیان محاسبه می‌شود.
اگر مقدار تابع یا گرادیان NaN باشد، تابع با کد خطا خاتمه می‌یابد.
آغاز جستجوی افزایشی (Expansion):

تا زمانی که مشتق جهت منفی است (db < 0.0)، مقدار گام افزایش می‌یابد.
اگر مقدار تابع از آستانه مجاز بیشتر شود، وارد فاز انقباض (cg_contract) می‌شود.
اگر تعداد تلاش‌ها زیاد شود یا شرایط خاصی رخ دهد، با کد خطا خارج می‌شود.
مقدار گام با توجه به پارامترهای الگوریتم و شرایط فعلی به‌روزرسانی می‌شود.
فاز انقباض (Line):

اگر در فاز قبل به نقطه‌ای برسیم که مشتق مثبت شود یا مقدار تابع از آستانه کمتر شود، وارد فاز انقباض می‌شویم.
در این فاز با استفاده از ترکیب روش‌های cubic، secant و bisection، بازه [a, b] کوچک می‌شود تا شرایط Wolfe برقرار شود.
در هر مرحله مقدار جدید گام محاسبه و ارزیابی می‌شود.
اگر شرایط Wolfe برقرار شود، تابع با موفقیت خاتمه می‌یابد.
اگر تعداد تلاش‌ها زیاد شود یا بازه بیش از حد کوچک شود، با کد خطا خارج می‌شود.
خروج:

وضعیت نهایی (کد موفقیت یا خطا) بازگردانده می‌شود.
نکات کلیدی:

این تابع هم فاز افزایش گام (expansion) و هم فاز انقباض (contraction) را مدیریت می‌کند.
از ترکیب چند روش (cubic, secant, bisection) برای انتخاب گام استفاده می‌شود تا همگرایی سریع‌تر و پایدارتر باشد.
شرایط Wolfe (یا Wolfe تقریبی) در هر مرحله بررسی می‌شود.
مدیریت خطا و شرایط خاص (مانند NaN، تعداد تلاش زیاد، یا عدم بهبود) به‌دقت انجام شده است.
این تابع قلب جستجوی خطی Wolfe در این پیاده‌سازی است و تضمین می‌کند که مقدار گام بهینه و قابل قبول برای الگوریتم بهینه‌سازی انتخاب شود.

````python
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


````
تابع cg_Wolfe(alpha, f, dphi, Com) بررسی می‌کند که آیا شرایط Wolfe یا Wolfe تقریبی برای مقدار گام فعلی برقرار است یا نه.

توضیح عملکرد:

اگر مقدار مشتق جهت (dphi) بزرگ‌تر یا مساوی حد پایین Wolfe (Com.wolfe_lo) باشد:
اگر اختلاف مقدار تابع (f - Com.f0) کمتر یا مساوی alpha * Com.wolfe_hi باشد:
اگر سطح چاپ ۲ یا بیشتر باشد، پیام "Wolfe conditions hold" چاپ می‌شود.
مقدار ۱ بازگردانده می‌شود (یعنی شرایط Wolfe برقرار است).
اگر Wolfe تقریبی فعال باشد (Com.AWolfe):
اگر مقدار تابع از آستانه مجاز (Com.fpert) کمتر یا مساوی باشد و مشتق جهت نیز از حد بالای Wolfe تقریبی (Com.awolfe_hi) کمتر یا مساوی باشد:
اگر سطح چاپ ۲ یا بیشتر باشد، پیام "Approximate Wolfe conditions hold" چاپ می‌شود.
مقدار ۱ بازگردانده می‌شود (یعنی شرایط Wolfe تقریبی برقرار است).
در غیر این صورت، مقدار ۰ بازگردانده می‌شود (شرایط برقرار نیست).
این تابع برای کنترل صحت گام انتخاب‌شده در جستجوی خطی Wolfe استفاده می‌شود.

````python
def line_search(x, n, dir, Stat, UParm, value, grad, valgrad):
    
    while True:    
        # Parm = lsu.CGParameter
        ParmStruct = lsu.CGParameter()
        Com = ls.CGCom
        
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
            # gnorm = max(gnorm, abs(t))
            d[i] = dir[i]
            dnorm2 += d[i] * d[i]

        for i in range (n5, n, 5):
            for j in range (5):
                t = g[i]
                gnorm2 += t * t
                if gnorm < abs (t): gnorm = abs (t)
                # gnorm = max(gnorm, abs(t))
                d[i] = dir[i]
                dnorm2 += d[i] * d[i]
            
        
            
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
        
        
    if exit:   
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
        

````
تابع line_search
این تابع هسته اصلی جستجوی خطی Wolfe است و وظیفه پیدا کردن گام بهینه (alpha) را در جهت کاهش مقدار تابع هدف بر عهده دارد. این تابع معمولاً در روش‌های بهینه‌سازی مبتنی بر گرادیان (مانند گرادیان مزدوج یا نزولی) استفاده می‌شود.

ورودی‌ها:
x: بردار متغیرها (نقطه فعلی)
n: تعداد متغیرها
dir: بردار جهت حرکت (معمولاً منفی گرادیان)
Stat: شیء آمارگر برای ذخیره نتایج (اختیاری)
UParm: پارامترهای کاربر (اختیاری)
value: تابع هدف
grad: تابع گرادیان
valgrad: تابعی که همزمان مقدار تابع هدف و گرادیان را می‌دهد
گام به گام عملکرد:
مقداردهی اولیه پارامترها و ساختارها
اگر پارامترهای کاربر داده نشده باشد، پارامترهای پیش‌فرض مقداردهی می‌شوند.
ساختار ارتباطی (Com) و آرایه‌های کاری (work) ساخته و تقسیم‌بندی می‌شوند.

محاسبه مقدار اولیه تابع هدف و گرادیان
مقدار اولیه تابع هدف و گرادیان در نقطه شروع محاسبه می‌شود.
همچنین نُرم x و گرادیان و جهت حرکت محاسبه می‌شود.

بررسی مقدار اولیه تابع هدف
اگر مقدار اولیه تابع هدف NaN باشد، تابع با خطا خاتمه می‌یابد.

بررسی شرط توقف اولیه
اگر نُرم گرادیان از آستانه تحمل کمتر باشد، الگوریتم متوقف می‌شود (همگرایی).

بررسی جهت نزولی بودن
اگر جهت حرکت، جهت نزولی نباشد (ضرب داخلی گرادیان و جهت مثبت باشد)، خطا برمی‌گرداند.

تخمین اولیه گام (alpha)
مقدار اولیه گام بر اساس نُرم x و گرادیان تخمین زده می‌شود.

بررسی امکان استفاده از گام درجه دوم یا سوم
اگر شرایط مناسب باشد، از گام درجه دوم (Quadratic) یا سوم (Cubic) برای تخمین بهتر alpha استفاده می‌شود.

به‌روزرسانی پارامترهای Wolfe
پارامترهای مربوط به شرایط Wolfe و Wolfe تقریبی مقداردهی می‌شوند.

اجرای جستجوی خطی اصلی
تابع cg_line فراخوانی می‌شود تا مقدار بهینه گام پیدا شود.

در صورت شکست Wolfe معمولی، Wolfe تقریبی امتحان می‌شود
اگر جستجوی خطی معمولی موفق نشود، حالت Wolfe تقریبی فعال و دوباره تلاش می‌شود.

کپی کردن مقدار جدید x
مقدار به‌دست‌آمده x (نقطه جدید) در آرایه x کپی می‌شود.

بررسی NaN شدن مقدار تابع یا گرادیان
اگر مقدار تابع یا گرادیان NaN شود، خطا برمی‌گرداند.

بررسی شرط توقف بر اساس تغییر تابع هدف
اگر تغییر تابع هدف خیلی کوچک باشد، الگوریتم متوقف می‌شود.

ذخیره نتایج در شیء آمارگر (Stat)
اگر شیء آمارگر داده شده باشد، نتایج نهایی (مقدار تابع، گرادیان، تعداد ارزیابی‌ها و ...) در آن ذخیره می‌شود.

چاپ پیام وضعیت نهایی
بسته به مقدار وضعیت (status)، پیام مناسب چاپ می‌شود (همگرایی، خطا، عدم بهبود، و ...).

خروج با کد وضعیت
مقدار نهایی status بازگردانده می‌شود.

## 2. line_search_user.py

### توضیح کلی  
این فایل ساختارها و پارامترهای مورد نیاز کاربر برای الگوریتم جستجوی خطی Wolfe را تعریف می‌کند. شامل کلاس‌هایی برای نگهداری پارامترهای تنظیمی الگوریتم (مانند دقت، تعداد تکرار، و غیره) و همچنین ساختارهایی برای ذخیره وضعیت فعلی حل (مانند مقدار تابع هدف، گرادیان، تعداد ارزیابی‌ها و غیره) است. این فایل به کاربر اجازه می‌دهد تا تنظیمات دلخواه خود را برای الگوریتم اعمال کند.

### کد کامل فایل

````python
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
        self.f = 0.0
        self.gnorm = 0.0
        self.nfunc = 0 
        self.ngrad = 0 
        self.g = None 
        self.alpha = 1.0

````
### توضیحات بخشی به بخش کد

## 3. line_search.py

### توضیح کلی  
این فایل ساختار داده‌ها و متغیرهای مورد نیاز برای اجرای جستجوی خطی Wolfe را تعریف می‌کند. شامل کلاس‌هایی برای نگهداری اطلاعات مربوط به وضعیت فعلی الگوریتم (مانند مقدار فعلی متغیرها، گرادیان، مقدار تابع هدف، و غیره) است. همچنین ارتباط بین پارامترهای کاربر و توابع الگوریتم را برقرار می‌کند و به عنوان واسط بین بخش کاربر و هسته الگوریتم عمل می‌کند.

### کد کامل فایل

````python
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
````
### توضیحات بخشی به بخش کد

## 4. gradient_method.py

### توضیح کلی  
این فایل پیاده‌سازی روش گرادیان نزولی (Gradient Descent) را ارائه می‌دهد که با استفاده از جستجوی خطی Wolfe، بهینه‌سازی تابع هدف را انجام می‌دهد. در این فایل، تابع هدف و گرادیان آن تعریف شده و با استفاده از توابع جستجوی خطی، مقدار بهینه متغیرها به دست می‌آید. این فایل نمونه‌ای از کاربرد عملی الگوریتم جستجوی خطی Wolfe در حل مسائل بهینه‌سازی است.


### کد کامل فایل

````python
import numpy as np
import line_search_user as lsu
import line_search_algorithm as lsa
import msvcrt

    
def InnerProduct(v, u, n):
    ip = 0.0
    for i in range(n):
        ip += v[i] * u[i]
    return ip
    # return np.dot(v, u)

# def cg_step(xtemp, x, d, alpha):
#     xtemp[:] = x + alpha * d

def myvalue(x, n):
    f = 0.0
    for i in range(n):
        t = i + 1
        t = np.sqrt(t)
        f += np.exp(x[i]) - t * x[i]
    return f

def mygrad(g, x, n):
    for i in range(n):
        t = i + 1
        t = np.sqrt(t)
        g[i] = np.exp(x[i]) - t
    return

def myvalgrad(g, x, n):
    f = 0.0
    for i in range(n):
        t = i + 1
        t = np.sqrt(t)
        ex = np.exp(x[i])
        f += ex - t * x[i]
        g[i] = ex - t
    return f



def gradient_method():
    cg_stat = lsu.CGStats()
    n = 100
    x = np.zeros(n)
    d = np.zeros(n)
    g = np.zeros(n)
    cg_stat.g = np.zeros(n)
    
    iter = 0
    
    for i in range(n):
        x[i] = 1.0
    # x[:] = [1.0] * len(x)
    
    fx = fx0 = myvalue(x, n)
    mygrad(g, x, n)
    
    for i in range(n):
        d[i] = -g[i]
    
    while True:
        iter += 1
        print(f"d'g = {InnerProduct(g, d, n)}\n")
          
        lsa.line_search(x, n, d, cg_stat, None, myvalue, mygrad, myvalgrad)
        print(f"line search:\n      nf = {cg_stat.nfunc},\n      ng = {cg_stat.ngrad}\n      alpha = {cg_stat.alpha}")
      
        for i in range(n):
            g[i] = cg_stat.g[i]
            
        fx0 = fx
        fx = cg_stat.f
        
        print(f"iter = {iter:<5d}, fxold = {fx0:10.6e}, fx ={fx:10.6e}, fxold-fx = {fx0 - fx:10.6e}\n")

        for i in range(n):
            d[i] = -g[i]
            
        gnorm2 = InnerProduct(g, g, n)
        print(f"gnorm2 = {gnorm2:10.6e}")
        
        
        if np.isnan(fx) or np.isinf(fx) or np.isnan(gnorm2) or np.isinf(gnorm2):
            print("NaN or Inf detected! Stopping.")
            break
        
        if gnorm2 <= 1e-8:
            break
        
        
    print(f"\nPlease press Enter to exit ")
    # msvcrt.getch().decode()
    input() 
    
if __name__ == "__main__":
    gradient_method()        
            

````

### توضیحات بخش به بخش 

````python
def InnerProduct(v, u, n):
    ip = 0.0
    for i in range(n):
        ip += v[i] * u[i]
    return ip

````
تابع InnerProduct(v, u, n) ضرب داخلی (Inner Product یا Dot Product) دو بردار را محاسبه می‌کند. این تابع سه ورودی دارد:

v و u: دو بردار (لیست یا آرایه) با طول n
n: تعداد عناصر هر بردار
درون تابع، یک متغیر ip برای جمع کردن حاصل ضرب هر جفت عنصر متناظر از دو بردار تعریف شده است. با یک حلقه for از ۰ تا n-1، هر بار مقدار v[i] * u[i] به ip اضافه می‌شود. در پایان، مقدار نهایی ip که همان ضرب داخلی دو بردار است، بازگردانده می‌شود.
ضرب داخلی معمولاً برای محاسبه زاویه بین دو بردار، طول بردار، یا در الگوریتم‌های بهینه‌سازی و یادگیری ماشین کاربرد دارد.

````python
def myvalue(x, n):
    f = 0.0
    for i in range(n):
        t = i + 1
        t = np.sqrt(t)
        f += np.exp(x[i]) - t * x[i]
    return f

````
تابع myvalue(x, n) مقدار یک تابع هدف را برای بهینه‌سازی محاسبه می‌کند.
ورودی‌ها:

x: یک بردار (لیست یا آرایه) با طول n
n: تعداد عناصر بردار
در این تابع:

متغیر f برای جمع کردن مقدار تابع هدف تعریف می‌شود.
با یک حلقه از ۰ تا n-1:
مقدار t برابر با ریشه دوم (sqrt) عدد i+1 محاسبه می‌شود.
سپس np.exp(x[i]) - t * x[i] به f اضافه می‌شود.
در پایان مقدار نهایی f بازگردانده می‌شود.
این تابع در واقع مجموع عبارت exp(x[i]) - sqrt(i+1) * x[i] را برای تمام عناصر بردار x محاسبه می‌کند.
این نوع توابع معمولاً به عنوان تابع هدف در مسائل بهینه‌سازی عددی استفاده می‌شوند.

````python
def mygrad(g, x, n):
    for i in range(n):
        t = i + 1
        t = np.sqrt(t)
        g[i] = np.exp(x[i]) - t
    return

````
تابع mygrad(g, x, n) گرادیان تابع هدف را برای هر عنصر از بردار x محاسبه می‌کند و نتیجه را در آرایه g قرار می‌دهد.

ورودی‌ها:

g: آرایه‌ای که مقدار گرادیان در آن ذخیره می‌شود.
x: بردار ورودی با طول n.
n: تعداد عناصر بردار.
در این تابع:

برای هر اندیس i از ۰ تا n-1:
مقدار t برابر با ریشه دوم (sqrt) عدد i+1 محاسبه می‌شود.
مقدار گرادیان برای هر عنصر به صورت np.exp(x[i]) - t محاسبه و در g[i] ذخیره می‌شود.
این تابع مشتق تابع هدف نسبت به هر متغیر را محاسبه می‌کند و در مسائل بهینه‌سازی برای تعیین جهت حرکت استفاده می‌شود.

````python
def myvalgrad(g, x, n):
    f = 0.0
    for i in range(n):
        t = i + 1
        t = np.sqrt(t)
        ex = np.exp(x[i])
        f += ex - t * x[i]
        g[i] = ex - t
    return f

````
تابع myvalgrad(g, x, n) همزمان مقدار تابع هدف و گرادیان آن را برای یک بردار ورودی محاسبه می‌کند.

ورودی‌ها:

g: آرایه‌ای که مقدار گرادیان در آن ذخیره می‌شود.
x: بردار ورودی با طول n.
n: تعداد عناصر بردار.
در این تابع:

متغیر f برای جمع کردن مقدار تابع هدف تعریف می‌شود.
برای هر اندیس i از ۰ تا n-1:
مقدار t برابر با ریشه دوم (sqrt) عدد i+1 محاسبه می‌شود.
مقدار ex برابر با np.exp(x[i]) محاسبه می‌شود.
مقدار تابع هدف با f += ex - t * x[i] به‌روزرسانی می‌شود.
مقدار گرادیان با g[i] = ex - t محاسبه و ذخیره می‌شود.
در پایان مقدار نهایی تابع هدف (f) بازگردانده می‌شود.
این تابع برای بهینه‌سازی‌هایی که همزمان به مقدار تابع هدف و گرادیان نیاز دارند، بسیار کاربردی است.

````python
def gradient_method():
    cg_stat = lsu.CGStats()
    n = 100
    x = np.zeros(n)
    d = np.zeros(n)
    g = np.zeros(n)
    cg_stat.g = np.zeros(n)
    
    iter = 0
    
    for i in range(n):
        x[i] = 1.0
    # x[:] = [1.0] * len(x)
    
    fx = fx0 = myvalue(x, n)
    mygrad(g, x, n)
    
    for i in range(n):
        d[i] = -g[i]
    
    while True:
        iter += 1
        print(f"d'g = {InnerProduct(g, d, n)}\n")
          
        lsa.line_search(x, n, d, cg_stat, None, myvalue, mygrad, myvalgrad)
        print(f"line search:\n      nf = {cg_stat.nfunc},\n      ng = {cg_stat.ngrad}\n      alpha = {cg_stat.alpha}")
      
        for i in range(n):
            g[i] = cg_stat.g[i]
            
        fx0 = fx
        fx = cg_stat.f
        
        print(f"iter = {iter:<5d}, fxold = {fx0:10.6e}, fx ={fx:10.6e}, fxold-fx = {fx0 - fx:10.6e}\n")

        for i in range(n):
            d[i] = -g[i]
            
        gnorm2 = InnerProduct(g, g, n)
        print(f"gnorm2 = {gnorm2:10.6e}")
        
        
        if np.isnan(fx) or np.isinf(fx) or np.isnan(gnorm2) or np.isinf(gnorm2):
            print("NaN or Inf detected! Stopping.")
            break
        
        if gnorm2 <= 1e-8:
            break
        
        
    print(f"\nPlease press Enter to exit ")
    # msvcrt.getch().decode()
    input() 

````
تابع gradient_method یک پیاده‌سازی از روش گرادیان نزولی (Gradient Descent) است که از جستجوی خطی Wolfe برای تعیین گام بهینه استفاده می‌کند.

توضیح گام به گام:

تعریف متغیرها و مقداردهی اولیه:

یک شیء آمارگر الگوریتم (cg_stat) ساخته می‌شود.
تعداد متغیرها (n) برابر 100 است.
بردارهای x (متغیرها)، d (جهت حرکت)، g (گرادیان) و cg_stat.g همگی با صفر مقداردهی اولیه می‌شوند.
مقدار اولیه همه عناصر x برابر 1 قرار داده می‌شود.
محاسبه مقدار اولیه تابع هدف و گرادیان:

مقدار تابع هدف (fx و fx0) با تابع myvalue محاسبه می‌شود.
گرادیان اولیه با تابع mygrad محاسبه و در g ذخیره می‌شود.
جهت حرکت اولیه d برابر منفی گرادیان قرار داده می‌شود (حرکت در جهت کاهش تابع هدف).
حلقه اصلی بهینه‌سازی:

شمارنده تکرارها (iter) افزایش می‌یابد.
مقدار ضرب داخلی گرادیان و جهت حرکت چاپ می‌شود.
جستجوی خطی Wolfe با تابع lsa.line_search انجام می‌شود تا مقدار بهینه گام (alpha) پیدا شود.
آمار جستجوی خطی (تعداد ارزیابی تابع و گرادیان، مقدار alpha) چاپ می‌شود.
گرادیان جدید از خروجی جستجوی خطی خوانده می‌شود.
مقدار تابع هدف به‌روزرسانی می‌شود و تغییر مقدار آن چاپ می‌شود.
جهت حرکت جدید دوباره برابر منفی گرادیان قرار می‌گیرد.
نُرم مربع گرادیان (gnorm2) محاسبه و چاپ می‌شود.
اگر مقدار تابع هدف یا نُرم گرادیان NaN یا Inf شود، یا نُرم گرادیان خیلی کوچک شود (شرط همگرایی)، حلقه متوقف می‌شود.
پایان:

پیام پایان چاپ می‌شود و برنامه منتظر فشردن Enter می‌ماند.
این تابع نمونه‌ای از یک الگوریتم بهینه‌سازی عددی با استفاده از گرادیان نزولی و جستجوی خطی Wolfe است.
