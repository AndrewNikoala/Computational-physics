import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

L = 1
N = 20
M = 1000
x1 = 0
x2 = L**2
t1 = 0
t2 = 1

x = np.linspace(x1, x2, N)
t = np.linspace(t1, t2, M)

h = x[1] - x[0]
tau = h**2/2 

u = np.zeros(M*N).reshape(M,N)
u[0] = x * (1 - x/(L**2))
print(u[0])

a = [0]
for i in range(1, N):
	a.append(-tau / (2 * h**2))

b = []
for i in range(0, N):
	b.append(1 + tau / (h**2))
	
c = []
for i in range(0, N-1):
	c.append(-tau / (2 * h**2))
c.append(0.)
temp = []
for m in range(0, M-1):

	d = []
	for i in range(N-1):
		d.append(u[m, i] + (tau / 2) * ((u[m, i+1] - 2*u[m, i] + u[m, i-1])/(h**2)))
	#d[0] = 0
	#d[N-2] = 0

	for i in range(1, N-1):
		xi = a[i]/b[i-1]
		a[i] = 0
		b[i] -= xi * c[i-1]
		d[i] -= xi * d[i-1]

	u[m+1, N-1] = d[N-2]/b[N-1]   
		
	for i in range(N-2, -1, -1):
		u[m+1, i] = 1/b[i] * (d[i] - c[i]*u[m, i+1])
	u[m+1, 0] = 0
	u[m+1, N-1] = 0

	temp.append(u[m+1].max())

'''
fig, ax = plt.subplots()
ln, = plt.plot([], [])
j = 0

def init():
	ax.set_xlim(x1, x2)
	ax.set_ylim(0, u.max())
	return ln,

def update(frame):
	ln.set_data(x, u[frame])
	return ln,

ani = FuncAnimation(fig, update, frames=M, init_func=init, blit=True)
plt.show()
'''

'''

plt.plot(x, u[0])
plt.plot(x, u[1])
plt.plot(x, u[2])
plt.plot(x, u[3])
plt.plot(x, u[4])
plt.show()
'''
fig, axs = plt.subplots()
axs.plot(temp)
axs.set_xlim(0, 100)

plt.show()
