import math
import matplotlib.pyplot as plt
import numpy as np
import time

# The beginning equation:
# du(x) / dx = -u(x), u(0) = 1, 0 < x < 3
# u = exp(-x)

def func(u):
    return -u

def func_eiler(y, i, h):
    return y[i - 1] + h * func(y[i - 1])

#def func_implicit_eiler(y, i, h, lambd):
#    return y[i - 1] / (1 + h * lambd)

def func_rk2(y, i, h, a):
    return y[i - 1] + h * ((1 - a) * func(y[i - 1]) + a * func(y[i - 1] + h * func(y[i - 1]) / (2 * a)))

def func_rk4(y, i, h):
    k1 = func(y[i - 1])
    k2 = func(y[i - 1] + h * k1 / 2)
    k3 = func(y[i - 1] + h * k2 / 2)
    k4 = func(y[i - 1] + h * k3 / 2)
    return y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

x_left = 0
x_right = 3.
u_0 = 1.
a = 3. / 4

print("Input precision n:")
N = int(input())
delta_x = (x_right - x_left) / N
y_e = np.zeros(N)
y_rk2 = np.zeros(N)
y_rk4 = np.zeros(N)
res = np.zeros(N)
x = np.zeros(N)

y_e[0] = u_0
y_rk2[0] = u_0
y_rk4[0] = u_0
x[0] = 0
res[0] = u_0
for i in range(1, N):
    x[i] = i * delta_x
    #Eiler`s method
    y_e[i] = func_eiler(y_e, i, delta_x)
    #Runge-Kutta method of second order
    y_rk2[i] = func_rk2(y_rk2, i, delta_x, a)
    #Runge-Kutta method of fourth order
    y_rk4[i] = func_rk4(y_rk4, i, delta_x)
    #Origin function with 100% precision
    res[i] = np.exp(-x[i])

fig, axs = plt.subplots(3, 1, figsize=(12, 10))
fig.subplots_adjust(hspace=0.5)

#axs[0].plot(x, y_e, "ro", x, res, "-")
axs[0].plot(x, abs(y_e - res))
axs[0].legend(["Euler`s method", "exp(-x)"])
axs[0].set_xlabel('x', fontsize = 15, color = 'blue')
axs[0].set_ylabel('y', fontsize = 15, color = 'blue')
axs[0].set_title('Euler`s method')
axs[0].grid()

#axs[1].plot(x, y_rk2, "ro", x, res, "-")
axs[1].plot(x, abs(y_rk2 - res))
axs[1].legend(["Runge-Kutta 2", "exp(-x)"])
axs[1].set_xlabel('x', fontsize = 12, color = 'blue')
axs[1].set_ylabel('y', fontsize = 12, color = 'blue')
axs[1].set_title('Runge-Kutta 2 method')
axs[1].grid()

#axs[2].plot(x, y_rk2, "ro", x, res, "-")
axs[2].plot(x, abs(y_rk4 - res))
axs[2].legend(["Runge-Kutta 4", "exp(-x)"])
axs[2].set_xlabel('x', fontsize = 12, color = 'blue')
axs[2].set_ylabel('y', fontsize = 12, color = 'blue')
axs[2].set_title('Runge-Kutta 4 method')
axs[2].grid()

plt.show()

    
#   Comparison of explicit Euler`s method with implicit Euler`s method    
    #x = np.zeros(N)
    #y_explicit_e = np.zeros(N)
    #y_implicit_e = np.zeros(N)
    #res = np.zeros(N)
    
    #x[0] = 0
    #y_explicit_e[0] = u_0
    #y_implicit_e[0] = u_0
    #res[0] = 1
    #lambd = 1
    
    #start_time = 0
    #end_time = 0
    
    #for i in range(1, N):
        #x[i] = i * delta_x
        #Origin function with 100% precision
        #res[i] = math.exp(-x[i])
    
    #start_time = time.time()
    #for i in range(1, N):
        #y_explicit_e[i] = func_eiler(y_explicit_e, i, delta_x)
    #end_time = time.time()
    #explicit_euler_time = end_time - start_time
    
    #start_time = time.time()
    #for i in range(1, N):
        #y_implicit_e[i] = func_implicit_eiler(y_implicit_e, i, delta_x, lambd)
    #end_time = time.time()
    #implicit_euler_time = end_time - start_time
    
    #fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    #fig.subplots_adjust(hspace=0.5)
    
    #axs[0].plot(x, y_explicit_e, "ro", x, res, "-")
    #axs[0].legend(["Explicit Eiler`s method", "exp(-x)"])
    #axs[0].set_xlabel('x', fontsize = 15, color = 'blue')
    #axs[0].set_ylabel('y', fontsize = 15, color = 'blue')
    #axs[0].set_title('Explicit Eiler`s method')
    #axs[0].grid()
    
    #axs[1].plot(x, y_implicit_e, "ro", x, res, "-")
    #axs[1].legend(["Implicit Eiler`s method", "exp(-x)"])
    #axs[1].set_xlabel('x', fontsize = 15, color = 'blue')
    #axs[1].set_ylabel('y', fontsize = 15, color = 'blue')
    #axs[1].set_title('Implicit Eiler`s method')
    #axs[1].grid()
    
    #plt.show()
    
    #print("Explicit time is :", explicit_euler_time)
    #print("Implicit time is :", implicit_euler_time)
