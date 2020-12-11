from math import *
import numpy as np
import matplotlib.pyplot as plt

def U(x):
    return x**2/2 + 10
    #return 0
n = 200
N = 6
x1 = -10
x2 = 10

m = 3

y0 = np.ones(n) + np.linspace(0, 1, n)

x = np.linspace(x1, x2, n, True)

h = x[1] - x[0]
#print(h)

a = [0]
for i in range(1, n):
    a.append(-1./(2*h**2))

b = []
for i in range(0, n):
    b.append(1./h**2+U(x[i]))
    
c = []
for i in range(0, n-1):
    c.append(-1./(2*h**2))
c.append(0.)
#print(a,b,c)
def TridiagMatrixAlg(a, b, c, d, n):
    y = []
    for i in range(0, n):
        y.append(0)
 
    for i in range(1, n):
        xi = a[i]/b[i-1]
        a[i] = 0
        b[i] -= xi * c[i-1]
        d[i] -= xi * d[i-1]
    
    y[n-1] = d[n-1]/b[n-1]    
        
    for i in range(n-2, -1, -1):
        y[i] = 1/b[i] * (d[i] - c[i]*y[i+1])
    
    return y

def InverseIterations(y0, a, b, c, n, N, m):
    psi = []
    E = []
    for j in range(m):
        d2 = y0.copy()
        for k in range(j):
            d2 = d2 - psi[k]*(np.inner(y0, psi[k]))/np.linalg.norm(psi[k])
        for i in range(N):
            d1 = d2
            d = d1.copy()
            a1 = a.copy()
            b1 = b.copy()
            c1 = c.copy()
            d2 = TridiagMatrixAlg(a1, b1, c1, d, n)
            for k in range(0, j):
                d2 = d2 - psi[k]*(np.inner(d2, psi[k]))/np.linalg.norm(psi[k])
    
    
        E0 = np.linalg.norm(d1)/np.linalg.norm(d2)
        d2 /= np.linalg.norm(d2)
        E.append(E0)
        psi.append(d2)
    return [E, psi]



[E, psi] = InverseIterations(y0, a, b, c, n, N, m)

print (E)


for i in range(0, m):
    plt.plot(x, psi[i], label = 'psi' + str(i))
plt.legend()
plt.show()
