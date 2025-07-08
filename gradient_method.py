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
    for i in range(n-1):
        f += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return f

def mygrad(g, x, n):
    g[:] = 0.0
    for i in range(n-1):
        g[i] += -400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1 - x[i])
        g[i+1] += 200.0 * (x[i+1] - x[i]**2)
        
def myvalgrad(g, x, n):
    f = 0.0
    g[:] = 0.0
    for i in range(n-1):
        t1 = x[i+1] - x[i] ** 2
        t2 = 1 - x[i]
        f += 100.0 * t1**2 + t2**2
        g[i] += -400.0 * x[i] * t1 - 2.0 *t2
        g[i+1] += 200.0 * t1
    return f
        
def gradient_method():
    cg_stat = lsu.CGStats()
    n = 10
    x = np.zeros(n)
    d = np.zeros(n)
    g = np.zeros(n)
    cg_stat.g = np.zeros(n)
    
    iter = 0
    
    for i in range(n):
        x[i] = -1.2 if i % 2 == 0 else 1.0
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
        
        
        # if np.isnan(fx) or np.isinf(fx) or np.isnan(gnorm2) or np.isinf(gnorm2):
        #     print("NaN or Inf detected! Stopping.")
        #     break
        
        if gnorm2 <= 1e-8:
            break
        print("##################################################################")

        
    print(f"\nPlease press Enter to exit ")
    # msvcrt.getch().decode()
    input() 
    
if __name__ == "__main__":
    gradient_method()        
            