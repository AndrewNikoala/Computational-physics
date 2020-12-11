import numpy as np
import matplotlib.pyplot as plt
from math import pi

a0 = 1
a1 = 0.002
w0 = 5.1

w1 = 25.5
t0 = 0
T = 2 * pi
t1 = T

def f(t):
	return a0 * np.sin(w0 * t) + a1 * np.sin(w1 * t)

def rectangle(t, k):
	return np.hstack((np.ones(k), np.zeros(t.size - k)))
	#return np.ones(t.size)

def hanna(t):
	return 0.5*(1-np.cos(2*pi*t/T))

def DFT(x):
	n = np.arange(x.size)
	k = n.reshape((x.size, 1))
	M = np.exp(-2j * pi * k * n / x.size)
	return np.dot(M, x)

def main():
	n = 2**10
	t = np.linspace(t0, t1, n)
	w = np.linspace(0, 1/t1, n)

	#fft_rect = np.abs(DFT(rectangle(t, n) * f(t)))
	#fft_han = np.abs(DFT(hanna(t) * f(t)))

	fft_rect = np.abs(np.fft.fft(rectangle(t, n) * f(t)))
	fft_han = np.abs(np.fft.fft(hanna(t) * f(t)))

	fig, axs =  plt.subplots(3)

	axs[0].plot(t, rectangle(t, n) * f(t))
	axs[0].plot(t, hanna(t) * f(t))

	axs[1].plot(np.arange(int(fft_rect.size) - 1) + 1, fft_rect[1:])
	axs[1].set_yscale('log')
	axs[1].set_title('Rectangle window')

	axs[2].plot(np.arange(int(fft_han.size) - 1) + 1, fft_han[1:])
	axs[2].set_yscale('log')
	axs[2].set_title('Hanna window')
 
	plt.show()
	return

if __name__ == '__main__':
	main() 
