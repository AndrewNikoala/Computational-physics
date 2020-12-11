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
T = 1

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

def create_matrix(N, h, tau, u):
    """Creating of tridiagonal matrix and right part of the system"""   
    
    A = np.zeros((N, N))
    a = - tau / (2*(h**2))
    b = 1 + tau / (h**2)
    c = - tau / (2*(h**2))
    col_offset = 0
    
    F = np.zeros(N)
        
    for col in range(0, N):
        if col == 0:
            A.itemset((0,0), 1)
            F[0] = 0
        elif col == N-1:
            A.itemset((N-1, N-1), 1)
            F[N-1] = 0
        else:
            A.itemset((col, col_offset), a)
            A.itemset((col, col_offset+1), b)
            A.itemset((col, col_offset+2), c)
            col_offset += 1
            
            # f = v_j + tau(v_(j-1) - 2v_j + v_(j+1))/(2h^2)
            F[col] = u[col] + tau * (u[col-1] - 2*u[col] + u[col+1])/(2 * (h**2))
    return A, F

def solve_diffusion_equation(N, h, tau, x):
    """ Solving of differntial equation """
    u = [f1_t(x)]
    row_u = np.squeeze(u)
    
    for m in range(1, N):
        A, F = create_matrix(N, h, tau, row_u)
        u = np.concatenate((u, [Gauss_exception_method(A, F, N)]))
        row_u = u[m, :]
    return u

def get_maxT(Res, N):
    """ Get max Temperature for particular step tau """
    max_indxs = Res.argmax(1)
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
    
    print("The boundary of tau`s convergence {:.6f}".format(h**2 / 2))
    
    Res = solve_diffusion_equation(N, h, tau, x)
    
    max_T = get_maxT(Res, N)
    
    fig, axs = plt.subplots(figsize=(12, 10))
    
    axs.plot(t, max_T, '-')
    axs.set_xlabel('t', fontsize = 15, color = 'blue')
    axs.set_ylabel('max T', fontsize = 15, color = 'blue')
    axs.set_title('Solving of the dif equation', size=30)
    axs.grid()
    
    plt.show()
