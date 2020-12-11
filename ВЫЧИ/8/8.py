import math
import matplotlib.pyplot as plt
import numpy as np
import time

# The beginning system:
#  / du / dt =  998u + 1998v
# {                              
#  \ dv / dt = -999u - 1999v

def f_u(u, v):
    return 998 * u + 1998 * v

def f_v(u, v):
    return -999 * u - 1999 * v

def func_explicit_euler(h, n, u_0, v_0):
    u = np.zeros(n)
    v = np.zeros(n)
    u[0] = u_0
    v[0] = v_0
    
    for i in range(1, n):
        u[i] = u[i - 1] + h * f_u(u[i - 1], v[i - 1])
        v[i] = v[i - 1] + h * f_v(u[i - 1], v[i - 1])
    return u, v

def func_implicit_euler(h, n, u_0, v_0, lamb):
    u = np.zeros(n)
    v = np.zeros(n)
    u[0] = u_0
    v[0] = v_0
    
    E = np.matrix('1 0; 0 1')
    A = np.matrix('998 1998; -999 -1999')
    B = np.linalg.inv(E - h*A)
    for i in range(1, n):
        B = np.linalg.inv(E - h*A)
        u[i] = B.item((0,0)) * u[i-1] + B.item((0,1)) * v[i-1]
        v[i] = B.item((1,0)) * u[i-1] + B.item((1,1)) * v[i-1]
    return u, v

if __name__ == "__main__":
    u_0 = 3.
    v_0 = -2.
   
    lambda_1 = -1
    lambda_2 = -1000

    h = 1 / (2 * abs(lambda_2))
    n = 10000
    t = np.linspace(0.001, 0.001 + h * n, n)
    
    start_time = time.time()
    u_eeul, v_eeul = func_explicit_euler(h, n, u_0, v_0)
    end_time = time.time()
    explicit_euler_time = end_time - start_time
    
    start_time = time.time()
    u_ieul, v_ieul = func_implicit_euler(h, n, u_0, v_0, abs(lambda_1))
    end_time = time.time()
    implicit_euler_time = end_time - start_time
    
    print('explicit Euler time: {:.3f} | implicit Euler time: {:.3f}'.format(explicit_euler_time, implicit_euler_time))
    
    
    alpha = 0.99
    beta = 0.02
    analitic_u = alpha * 2 * np.exp(-t) + beta * 1 * np.exp(-t * 1000)
    analitic_v = alpha * (-1) * np.exp(-t) + beta * (-1) * np.exp(-t * 1000)
    
    #fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    #fig.subplots_adjust(hspace=0.5)
    
    #axs[0, 0].plot(t, np.abs(u_eeul - analitic_u))
    #axs[0, 0].set_xlabel('t', fontsize = 15, color = 'blue')
    #axs[0, 0].set_ylabel('du', fontsize = 15, color = 'blue')
    #axs[0, 0].set_title('Explicit Euler`s method')
    #axs[0, 0].set_xscale('log')
    #axs[0, 0].set_yscale('log')
    #axs[0, 0].grid()
    
    #axs[0, 1].plot(t, np.abs(v_eeul - analitic_v))
    #axs[0, 1].set_xlabel('t', fontsize = 15, color = 'blue')
    #axs[0, 1].set_ylabel('dv', fontsize = 15, color = 'blue')
    #axs[0, 1].set_title('Explicit Euler`s method')
    #axs[0, 1].set_xscale('log')
    #axs[0, 1].set_yscale('log')
    #axs[0, 1].grid()
    
    #axs[1, 0].plot(t, np.abs(u_ieul - analitic_u))
    #axs[1, 0].set_xlabel('t', fontsize = 15, color = 'blue')
    #axs[1, 0].set_ylabel('du', fontsize = 15, color = 'blue')
    #axs[1, 0].set_title('Implicit Euler`s method')
    #axs[1, 0].set_xscale('log')
    #axs[1, 0].set_yscale('log')
    #axs[1, 0].grid()
    
    #axs[1, 1].plot(t, np.abs(v_ieul - analitic_v))
    #axs[1, 1].set_xlabel('t', fontsize = 15, color = 'blue')
    #axs[1, 1].set_ylabel('dv', fontsize = 15, color = 'blue')
    #axs[1, 1].set_title('Implicit Euler`s method')
    #axs[1, 1].set_xscale('log')
    #axs[1, 1].set_yscale('log')
    #axs[1, 1].grid()
    
    #plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(t, u_eeul, '*', label = 'explicit euler')
    axs[0].plot(t, u_ieul, 'ro', label = 'implicit euler')
    axs[0].plot(t, analitic_u, ':', label = 'analitic')
    axs[0].legend(fontsize = 12,
          facecolor = 'oldlace',    #  цвет области
          edgecolor = 'r',    #  цвет крайней линии
         )
    axs[0].set_xlabel('t', fontsize = 15, color = 'blue')
    axs[0].set_ylabel('u', fontsize = 15, color = 'blue')
    axs[0].set_xlim(0, 0.01)
    axs[0].set_title('Plot for u(t)')
    axs[0].grid()
    
    axs[1].plot(t, v_eeul, '*', label = 'explicit euler')
    axs[1].plot(t, v_ieul, 'ro', label = 'implicit euler')
    axs[1].legend(fontsize = 12,
          facecolor = 'oldlace',    #  цвет области
          edgecolor = 'r',    #  цвет крайней линии
         )
    axs[1].plot(t, analitic_v, ':', label = 'analitic')
    axs[1].set_xlabel('t', fontsize = 15, color = 'blue')
    axs[1].set_ylabel('v', fontsize = 15, color = 'blue')
    axs[1].set_xlim(0, 0.01)
    axs[1].set_title('Plot for v(t)')
    axs[1].grid()
    
    plt.show()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5)
    
    axs[0].plot(u_eeul, v_eeul, 'ro')
    axs[0].set_xlabel('u_eeul', fontsize = 15, color = 'blue')
    axs[0].set_ylabel('v_eeul', fontsize = 15, color = 'blue')
    axs[0].set_title('Phase plot')
    axs[0].grid()
    
    axs[1].plot(u_ieul, v_ieul, 'ro')
    axs[1].set_xlabel('u_ieul', fontsize = 15, color = 'blue')
    axs[1].set_ylabel('v_ieul', fontsize = 15, color = 'blue')
    axs[1].set_title('Phase plot')
    axs[1].grid()

    plt.show()
