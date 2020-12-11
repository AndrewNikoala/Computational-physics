#!/usr/bin/python3.7

import numpy as np
import matplotlib.pyplot as plt

A = 998
B = 1998
C = -999
D = -1999

def f_x(x, y):
	return 998*x + 1998*y

def f_y(x, y):
	return -999*x-1999*y

def runge_kutta_4(h, n, x0, y0):
	x = np.zeros(n)
	y = np.zeros(n)
	x[0] = x0
	y[0] = y0
	k_x = np.zeros(4)
	k_y = np.zeros(4)
	for i in range(1, n):
		k_x[0] = f_x(x[i-1], y[i-1])
		k_y[0] = f_y(x[i-1], y[i-1])

		k_x[1] = f_x(x[i-1] + (h * k_x[1])/2, y[i-1] + (h * k_y[1])/2)
		k_y[1] = f_y(x[i-1] + (h * k_x[1])/2, y[i-1] + (h * k_y[1])/2)
		
		k_x[2] = f_x(x[i-1] + (h * k_x[2])/2, y[i-1] + (h * k_y[2])/2)
		k_y[2] = f_y(x[i-1] + (h * k_x[2])/2, y[i-1] + (h * k_y[2])/2)
		
		k_x[3] = f_x(x[i-1] + h * k_x[3], y[i-1] + h * k_y[3])
		k_y[3] = f_y(x[i-1] + h * k_x[3], y[i-1] + h * k_y[3])
		
		x[i] = x[i-1] + (k_x[0] + 2 * k_x[1] + 2 * k_x[2] + k_x[3]) * h / 6
		y[i] = y[i-1] + (k_y[0] + 2 * k_y[1] + 2 * k_y[2] + k_y[3]) * h / 6
	return x, y


def main():
    a = 0
    b = 10
    n = 3000
    t = np.linspace(a, b, n)
    u = np.zeros(n)
    v = np.zeros(n)
    alpha = 1
    beta = 0.001

    t[0] = 0
    u[0] = alpha * 2 + beta * 1
    v[0] = alpha * (-1) + beta * (-1)
    h = t[1] - t[0]

    for i in range(1, n):
        u[i] = ((1 - h * D)*u[i-1] + h * B*v[i-1]) / ((h * A - 1) * (h * D - 1) - h * h * B * C)
        v[i] = ((1 - h * A)*v[i-1] + h * C*u[i-1]) / ((h * A - 1) * (h * D - 1) - h * h * B * C)

    u_4, v_4 = runge_kutta_4(h, n, u[0], v[0])

    analitic_u = alpha * 2 * np.exp(-t) + beta * 1 * np.exp(-t*1000)
    analitic_v = alpha * (-1) * np.exp(-t) + beta * (-1) * np.exp(-t*1000)

    fig, axs = plt.subplots(2, 2)

    axs[0,0].plot(t, np.abs(u - analitic_u))
    axs[0,1].plot(t, np.abs(v - analitic_v))
    axs[1,0].plot(t, np.abs(u_4 - analitic_u))
    axs[1,1].plot(t, np.abs(v_4 - analitic_v))

    axs[0,0].set_title('Euler U error')
    axs[0,1].set_title('Euler V error')
    axs[1,0].set_title('R-K 4 U error')
    axs[1,1].set_title('R-K 4 V error')

    axs[0,0].set_yscale('log')
    axs[0,1].set_yscale('log')
    axs[1,0].set_yscale('log')
    axs[1,1].set_yscale('log')

    axs[0,0].set_xscale('log')
    axs[0,1].set_xscale('log')
    axs[1,0].set_xscale('log')
    axs[1,1].set_xscale('log')

    '''
    axs[0,2].plot(t, u)
    axs[0,3].plot(t, v)
    axs[1,2].plot(t, u_4)
    axs[1,3].plot(t, v_4)

    axs[0,2].plot(t, analitic_u)
    axs[0,3].plot(t, analitic_v)
    axs[1,2].plot(t, analitic_u)
    axs[1,3].plot(t, analitic_v)
    '''
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5)

    axs[0].plot(u_4, v_4)
    axs[0].set_xlabel('u', fontsize = 15, color = 'blue')
    axs[0].set_ylabel('v', fontsize = 15, color = 'blue')
    axs[0].set_title('Explicit Euler`s method')
    axs[0].grid()

    axs[1].plot(u, v)
    axs[1].set_xlabel('u', fontsize = 15, color = 'blue')
    axs[1].set_ylabel('v', fontsize = 15, color = 'blue')
    axs[1].set_title('Implicit Euler`s method')
    axs[1].grid()

    plt.show()

if __name__ == '__main__':
	main() 
