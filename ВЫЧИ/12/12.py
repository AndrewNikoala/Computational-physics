import numpy as np
import matplotlib.pyplot as plt
"""
    Сигнал, состоящий из двух гармонических осцилляций с различными частотами и амплитудами, 
    f(t) = a0*sin(w0*t) + a1*sin(w1*t) регистрируется на некотором интервале T.
    Вычислить и построить график спектра мощности. Сравнить спектры, полученные с прямоугольным окном 
    и окном Ханна, при следующих параметрах a0 = 1, a1 = 0.002, w0 = 5.1, w1=5w0 = 25.5, T = 2pi
""" 

"""
    Constants
"""
a_0 = 1
a_1 = 0.002
w_0 = 5.1
w_1 = 25.5
T = 2*np.pi

def f1(t):
    return a_0 * np.sin(w_0*t) + a_1 * np.sin(w_1*t)

def FourierTransformation(f_t, N):
    f_j = np.zeros(N, dtype=complex)
    C = T / N
    i = 1j
    for j in range(0, N):
        for k in range(0, N):
            f_j[j] += C * f_t[k] * np.exp(2*np.pi * i * j * k / N)
    #return f_j.real
    return abs(f_j)

def getDotsRW(N):
    """ Rectangular window """
    f_t = np.zeros(N, dtype=complex)
    tau = T / N
    for k in range(0, N):
        f_t[k] = f1(k * tau)
    return f_t

def getDotsHW(N):
    """ Hann window """
    f_t = np.zeros(N, dtype=complex)
    h = lambda k: (1 - np.cos(2*np.pi*k / N))/2
    tau = T / N
    for k in range(0, N):
        f_t[k] = f1(k * tau) * h(k)
    return f_t

def fill_w(N):
    w = np.zeros(N)
    for j in range(0, int(N/2)):
        w[j] = j*2*np.pi/T
    for j in range(int(N/2), N):
        w[j] = (-1 + j/N)*2*np.pi*N/T
    return w

if __name__ == "__main__":
    print("Input number of dots:")
    N = int(input())

    t = np.linspace(0, T, N)
    w = fill_w(N)
    
    f_tRW = getDotsRW(N)
    f_tHW = getDotsHW(N)
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5)
    
    axs[0].plot(t, f_tRW.real, '-', t, f_tHW.real, '-')
    axs[0].set_xlabel('t', fontsize = 15, color = 'blue')
    axs[0].set_ylabel('f_j', fontsize = 15, color = 'blue')
    axs[0].set_title('Rectangular window \nHann window', size=15)
    axs[0].grid()
    
    axs[1].plot(w, FourierTransformation(f_tRW, N), '-')
    #axs[1].plot(range(N), abs(np.fft.fft(f_tRW)), '-')
    axs[1].set_xlabel('w', fontsize = 15, color = 'blue')
    axs[1].set_ylabel('f_j', fontsize = 15, color = 'blue')
    axs[1].set_title('Rectangular window', size=15)
    axs[1].set_yscale('log')
    axs[1].set_xlim(-50, 50)
    axs[1].grid()
    
    axs[2].plot(w, FourierTransformation(f_tHW, N), '-')
    #axs[2].plot(range(N), abs(np.fft.fft(f_tHW)), '-')
    axs[2].set_xlabel('w', fontsize = 15, color = 'blue')
    axs[2].set_ylabel('f_j', fontsize = 15, color = 'blue')
    axs[2].set_title('Hann window', size=15)
    axs[2].set_yscale('log')
    axs[2].set_xlim(-50, 50)
    axs[2].grid()
    
    plt.show()
