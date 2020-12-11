"""
########## Вариант 10 ###########

Найти зависимость температуры от времени в центре двумерной квадратной области в задаче
с анизатропной теплопроводностью

du/dt = d^2 u/ dx^2 + 0.5 * d^2 u/d y^2

u(-L, y, t) = u(L, y, t) = sin(2pi*t)

u(x, -L, t) = u(x, L, t) = 0

L = 1

Замечание: tau < 0.5 / (1./h_x**2 + 1./(2h_y**2))
Если tau > этой величины, то наблюдаются большие шумы
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation
import numpy as np
import time

# Constants
L = 1

# Edge conditions
def edgeX(t):
    return np.sin(2*np.pi*t)

def edgeY(t):
    return 0

# Free function in heat equation
def f(t):
    return 0

"""
1) Явная схема
"""

X_0, X_J = (-L, L)
Y_0, Y_K = (-L, L)
t_0, t_M = (0, 4)

class SolutionGrid():

    def __init__(self, J, K, M):
        self.J = J - 1
        self.K = K - 1
        self.M = M - 1
        
        self.u = np.zeros((J, K, M))
        self.x = np.linspace(X_0, X_J, J)
        self.y = np.linspace(Y_0, Y_K, K)
        self.t = np.linspace(t_0, t_M, M)
        
        self.h_x = self.x[1] - self.x[0]
        self.h_y = self.y[1] - self.y[0]
        self.tau = self.t[1] - self.t[0]
        
        print(f'h_x: {self.h_x}')
        print(f'h_y: {self.h_y}')
        print(f'tau: {self.tau}')
        print(f'tau <  {(0.5 / ( 1./(self.h_x**2) + 1./(2*self.h_y**2)))}\n')
        
        self.anim = None
        self.centreTemp = np.zeros(M)
        
        self.a = self.tau / (self.h_x**2)
        self.b = self.tau / (2*self.h_y**2)
        self.c = 1 - self.tau*(2./(self.h_x**2) + 1./(self.h_y**2))
        self.solveEquation()
        self.compCentreTemp()
        
    def itemset(self, j, k, m, value):
        self.u[i][j][k] = value
        
    def itemget(self, j, k, m):
        return self.u[i][j][k]
        
    def getSliceTime(self, m):
        """
        Get solution of the equation for specific time
        """
        F = np.zeros((self.J + 1, self.K + 1))
        for j in range(0, self.J + 1):
            for k in range(0, self.K + 1):
                F.itemset((j, k), self.u[j][k][m])
        #print(F)
        return F
        
    def printSolution(self):
        print(self.u)

    def PointSolution(self, j, k, m):
        self.u[j][k][m+1] =\
            self.a*(self.u[j+1][k][m] + self.u[j-1][k][m]) +\
            self.b*(self.u[j][k+1][m] + self.u[j][k-1][m]) +\
            self.c*self.u[j][k][m] + f(m * self.tau) * self.tau
    
    def solveEquation(self):
        for j in range(0, self.J + 1):
            for m in range(0, self.M + 1):
                self.u[j][0][m] = edgeY(m * self.tau)
                self.u[j][self.K][m] = edgeY(m * self.tau)

        for k in range(0, self.K + 1):
            for m in range(0, self.M + 1):
                self.u[0][k][m] = edgeX(m * self.tau) 
                self.u[self.J][k][m] = edgeX(m * self.tau)

                
        for m in range(1, self.M):
            for k in range(1, self.K):
                for j in range(1, self.J):
                    self.PointSolution(j, k, m)

    def drawGifSolution(self):
        X, Y = np.meshgrid(self.x, self.y)
        def animate(m, plot, plot_args):
            ax.cla()
            ax.set_xlim3d([-L, L])
            ax.set_ylim3d([-L, L])
            ax.set_zlim3d([-1, 1])
            
            plot = ax.plot_surface(X, Y, self.getSliceTime(m), **plot_args)
            return plot,
        
        plot_args = {'cmap': cm.bwr, 'color': 'w'}
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([-L, L])
        ax.set_ylim3d([-L, L])
        ax.set_zlim3d([-1, 1])
        plot = ax.plot_surface(X, Y, self.getSliceTime(0), **plot_args)
        self.anim = FuncAnimation(fig, animate, frames=self.M + 1, fargs=(plot, plot_args),
                                interval=10, blit=False)
        #anim.save('animationHE1.gif', writer='pillow', fps=120)
        plt.show()
    
    def compCentreTemp(self):
        for m in range(0, self.M):
            self.centreTemp[m] = self.u[int(self.J/2)][int(self.K/2)][m]
    
    def drawCentreTemp(self):
        fig, axs = plt.subplots(1, 1, figsize=(12, 10))
        axs.plot(self.t, self.centreTemp)
        axs.set_xlabel('t', fontsize = 15, color = 'blue')
        axs.set_ylabel('T', fontsize = 15, color = 'blue')
        axs.set_title('Centre temperature', size=15)
        axs.grid()
        
        plt.show()
    
    def animationSave(self, gifName):
        self.anim.save(gifName, writer='pillow', fps=120)
