import numpy as np
import matplotlib.pyplot as plt
"""
    Решить задачу Коши для одномерного уравнения диффузии по схеме Кранка-Николсона
    du/dt = d^2(u)/(dx)^2 , 0 < x < L, L = 1
    u(0, t) = u(L, t) = 0, u(x, 0) = x(1 -x/L)^2
    
    На каждом шаге по времени найти максимальное значение температуры и нарисовать зависимость
    максимальной температуры от времени. Показать, что на больших временах она убывает экспоненциально
"""

L = 1
T = 100

def f1_t(x):
    return x * (1 - x / L)**2

def Gauss_exception_method(A, F, N):
    """Straight Gaussian move"""
    for row in range(0, N-1):
        ksi = A.item(row+1, row) / A.item(row, row) # ksi = a_i / b_(i-1)
        b = A.item(row+1, row+1)
        c = A.item(row, row+1)
        
        A.itemset((row+1, row), 0)           # a_i = 0
        A.itemset((row+1, row+1), b - ksi*c) # b_i = b_i - ksi*c_(i-1)
        
        F[row + 1] -= ksi*F[row]
    
    """Reverse Gaussian move"""
    y = np.zeros(N)
    y[N-1] = F[N-1] / A.item(N-1, N-1)  # y_n = f_n / b_n
    for row in range(N - 2, -1, -1):
        # y_i = (f_i - c_i * y_(i+1)) / b_i
        y[row] = (F[row] - A.item(row,row+1) * y[row+1]) / A.item(row,row)
    
    return y

def create_matrix(N, h, tau, jx):
    """Creating of tridiagonal matrix and right part of the system"""   
    
    A = np.zeros((N, N))
    a = - tau / (2*(h**2))
    b = 1 + tau / (h**2)
    c = - tau / (2*(h**2))
    col_offset = 0
    
    F = np.zeros(N)
    f = f1_t(jx * h) + tau * (f1_t((jx-1) * h) - 2*f1_t(jx * h) + f1_t((jx+1 * h))) \
        /(2 * (h**2)) # f = v_j + tau(v_(j-1) - 2v_j + v_(j+1))/(2h^2)
    
    for row in range(0, N):
        if row == 0:
            A.itemset((0,0), b)
            A.itemset((0,1), c)
            F[row] = 0
        elif row == N-1:
            A.itemset((N-1, N-2), a)
            A.itemset((N-1, N-1), b)
            F[row] = 0
        else:
            A.itemset((row, col_offset), a)
            A.itemset((row, col_offset+1), b)
            A.itemset((row, col_offset+2), c)
            col_offset += 1
            
            F[row] = f
    return A, F

def solve_diffusion_equation(N, h, tau):
    """ Solving of differntial equation """
    u = np.zeros((1,N))
    for jx in range(1, N - 1):
        A, F = create_matrix(N, h, tau, jx)
        u = np.concatenate((u, [Gauss_exception_method(A, F, N)]))
    u = np.concatenate((u, np.zeros((1, N))))

    return u

def get_maxT(Res, N):
    """ Get max Temperature for particular step tau """
    max_indxs = Res.argmax(0)
    max_T = np.zeros(N)
    for row in range(0, N):
        max_T[row] = Res.item((row, max_indxs[row]))
    return max_T
if __name__ == "__main__":
    print("Input number of steps:")
    N = int(input())
    
    x = np.linspace(0, L, N)
    t = np.linspace(0, T, N)
    h = x[1] - x[0]
    tau = t[1] - t[0]
    
    Res = solve_diffusion_equation(N, h, tau)
    
    max_T = get_maxT(Res, N)
    
    fig, axs = plt.subplots(figsize=(12, 10))
    
    axs.plot(t, max_T, 'ro', label = 'numerical solution')
    axs.set_xlabel('t', fontsize = 15, color = 'blue')
    axs.set_ylabel('max T', fontsize = 15, color = 'blue')
    axs.set_title('Solving of the dif equation', size=30)
    axs.grid()
    
    plt.show()
