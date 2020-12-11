import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def func(A, U, E):
    return (np.tan(np.sqrt(2 * (U - E)) / A) - np.sqrt(E / (U - E)))

def d_func(A, U, E):
    return -(1. / (math.sqrt(2 * (U - E)) * math.cos(math.sqrt(2 * (U - E) / A)) ** 2) + U / (2 * math.sqrt(E * (U - E) ** 3)))

def func1(A, U, E):
    return np.tan(E)

def d_func1(A, U, E):
    return 1. / (math.cos(E) ** 2)

def func2(E):
    return (E - 5) ** 2

print("Input A:")
A = float(input())
print("Input U:")
U = float(input())
print("Input precision p:")
p = float(input())
print()

#  The Bisection Method

# 1. Define range [a,b]
a = 1.
b = 0
E = 0
step = 0.1 # U should be more than step
i = 0
while b == 0 or a == 1:
    i = i + 1
    if U - E < 0:
        print("Boundaries are not found")
        print("Equation don`t have solution with these parameters")
        sys.exit()
    t = func(A, U, E)
    if E >= 0:
        if t > 0:
            b = E
        if t <= 0:
            a = E
    E = E + step
if (a > b):
    temp = a
    a = b
    b = temp

# 2. Method of dichotomy
i = 0
a = 6
b = 7
while(b - a >= p):
    if func1(A, U, a) * func1(A, U, (a + b) / 2) <= 0:
        b = (a + b) / 2
    else: 
        a = (a + b) / 2
    i = i + 1
E_res = (a + b) / 2
print("Bisection Method: E_0 = %.8f" % E_res, "||  i = ", i)


#  The Iteration Method

E_cur = 5
E_prev = p
lamb = 1. / d_func1(A, U, E_cur)
i = 0
while True:
    E_prev = E_cur
    E_cur = E_prev - lamb * func1(A, U, E_prev)
    if abs(E_cur - E_prev) < p:
        break
    i = i + 1
E_res = E_cur
print("Iteration Method: E_0 = %.8f" % E_res, "||  i = ", i)

# The Newton Method
E_cur = 5
E_prev = E_cur
lamb = 0
i = 0
while True:
    lamb = 1 / d_func1(A, U, E_prev)
    E_prev = E_cur
    E_cur = E_prev - lamb * func1(A, U, E_prev)
    i = i + 1
    if abs(E_cur - E_prev) < p:
        break
E_res = E_cur
print("Newton Method:    E_0 = %.8f" % E_res, "||  i = ", i)
