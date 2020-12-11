import math
import numpy as np
import matplotlib.pyplot as plt

N = 100
eps = 1e-10

a = 0
b = math.pi

def func(x, m, t):
    return math.cos(m * t - x * math.sin(t)) / math.pi  # (0, pi)
# calculating of Bessel`s Integral
def integrate(x, m):
    delta_t = (b - a) / N
    summa = 0
    for i in range(N):
        a_i = a + i * delta_t
        b_i = a + (i + 1) * delta_t
        summa = summa + (func(x, m, a_i) + 4 * func(x, m, (a_i + b_i) / 2) + func(x, m, b_i)) * delta_t / 6
    return summa
# derivative of Bessel`s function
def derivative_1(x, m, dx):
    #dx = 0.00001
    return (integrate(x + dx, m) - integrate(x - dx, m)) / (2 * dx)


if __name__ == '__main__':
    n = 50
    x_0 =  2 * math.pi / n
    
    delta_e = np.zeros(n)
    dx = np.zeros(n)
    rate_dx = 0.5
    dx[0] = 1.
    for i in range(0, n):
        if i != 0:
            dx[i] = dx[i-1] * rate_dx
        x = x_0 * i
        #res = derivative(x, 0, dx[i]) + integrate(x, 1)
        res1 = derivative_1(x, 0, dx[i])
        res2 = integrate(x, 1)
        delta_e[i] = abs(res2 + res1)
        #if res > eps:
        #    print()
        #    print("precision not enough", i)
        #    break
    fig, axs = plt.subplots(figsize=(12, 10))
    
    axs.plot(dx, delta_e)
    axs.set_xlabel('dx', fontsize = 15, color = 'blue')
    axs.set_ylabel('delta_e', fontsize = 15, color = 'blue')
    axs.set_title('e(dx)')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.grid()
    
    plt.show() 
