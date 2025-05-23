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

print()
````
### توضیحات بخشی به بخش کد

## 2. line_search_user.py

### توضیح کلی  
این فایل ساختارها و پارامترهای مورد نیاز کاربر برای الگوریتم جستجوی خطی Wolfe را تعریف می‌کند. شامل کلاس‌هایی برای نگهداری پارامترهای تنظیمی الگوریتم (مانند دقت، تعداد تکرار، و غیره) و همچنین ساختارهایی برای ذخیره وضعیت فعلی حل (مانند مقدار تابع هدف، گرادیان، تعداد ارزیابی‌ها و غیره) است. این فایل به کاربر اجازه می‌دهد تا تنظیمات دلخواه خود را برای الگوریتم اعمال کند.

### کد کامل فایل

````python

print()
````
### توضیحات بخشی به بخش کد

## 3. line_search.py

### توضیح کلی  
این فایل ساختار داده‌ها و متغیرهای مورد نیاز برای اجرای جستجوی خطی Wolfe را تعریف می‌کند. شامل کلاس‌هایی برای نگهداری اطلاعات مربوط به وضعیت فعلی الگوریتم (مانند مقدار فعلی متغیرها، گرادیان، مقدار تابع هدف، و غیره) است. همچنین ارتباط بین پارامترهای کاربر و توابع الگوریتم را برقرار می‌کند و به عنوان واسط بین بخش کاربر و هسته الگوریتم عمل می‌کند.

### کد کامل فایل

````python

print()
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
