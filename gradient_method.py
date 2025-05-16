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
        g[i] + ex - t
    return f

cg_stat = lsu.CGStats()

def gradient_method():
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
        
        if gnorm2 <= 1e-8:
            break
        
        
    print(f"\nPlease press Enter to exit ")
    msvcrt.getch().decode()
    # input()[0]
    
if __name__ == "__main__":
    gradient_method()        
            