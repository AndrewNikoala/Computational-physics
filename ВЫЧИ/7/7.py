import math
import matplotlib.pyplot as plt
import numpy as np

# The beginning system:
#  / dx / dt = a * x - b * x * y
# {                              , x, y > 0
#  \ dy / dt = c * x * y - d * y
#   a - birth rate,             b - kill of the victim
#   c - kill of the victim      d - death from starvation
#   a = 10 = d,   b = 2 = c

def f_x(x, y, a = 10 , b = 2):
    return a * x - b * x * y

def f_y(x, y, c = 2, d = 10):
    return c * x * y - d * y

def func_rk2(h, n, x_0, y_0, a):
    x = []
    y = []
    x.append(x_0)
    y.append(y_0)
    
    for i in range(1, n):
        x.append(x[i-1] + h * ((1 - a) * f_x(x[i-1], y[i-1]) + a * f_x(x[i-1] + (h * f_x(x[i-1], y[i-1]))/(2*a), y[i-1] + (h * f_y(x[i-1], y[i-1]))/(2*a))))
        y.append(y[i-1] + h * ((1 - a) * f_y(x[i-1], y[i-1]) + a * f_y(x[i-1] + (h * f_x(x[i-1], y[i-1]))/(2*a), y[i-1] + (h * f_y(x[i-1], y[i-1]))/(2*a))))
    return x, y


# x_0 = 5.01
# y_0 = 4.99
x_0 = 2
y_0 = 1


a = 10
b = 2
n = 1000

t = np.linspace(a, b, n)
h = t[1] - t[0]
alpha = 3. / 4

x, y = func_rk2(h, n, x_0, y_0, alpha)

fig, axs = plt.subplots()

axs.plot(x, y)
axs.set_xlabel('x', fontsize = 15, color = 'blue')
axs.set_ylabel('y', fontsize = 15, color = 'blue')
#axs.set_xlim(4, 6)
#axs.set_ylim(4, 6)
axs.set_title('Lotka–Volterra equations')
axs.grid()

plt.show() 
