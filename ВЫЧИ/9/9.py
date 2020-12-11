import numpy as np
import matplotlib.pyplot as plt
"""
    Методом прогонки решить:
    y''(x) = sin(x) , 0 < x < pi     // y(x) = -sin(x) + c1*x + c2
    
    Конечная разность: y''(x) = {y_(i-1) - 2y_i + y_(i+1)}/(h^2) = f_i, i=1..3
    
    Рассмотреть различные варианты граничных условий
"""

"""
    1) Граничные условия вида y(0) = y(pi) = y_0
"""

def f1(x):
    return np.sin(x)

def solv_func(x, y_0):
    c_2 = y_0
    c_1 = 0
    return -np.sin(x) + c_1*x + c_2
    
def create_matrix(N, h, y_0):
    """Creating of tridiagonal matrix and right part of the system"""   
    A = np.zeros((N, N))
    a = 1. / (h**2)
    b = -2. / (h**2)
    c = 1. / (h**2)
    col_offset = 0
    
    F = np.zeros(N)
    
    for raw in range(0, N):
        if raw == 0:
            A.itemset((0,0), 1)
            F[0] = y_0
        elif raw == N-1:
            A.itemset((N-1, N-1), 1)
            F[N-1] = y_0
        else:
            A.itemset((raw, col_offset), a)
            A.itemset((raw, col_offset+1), b)
            A.itemset((raw, col_offset+2), c)
            col_offset += 1

            F[raw] = f1(raw * h)
    return A, F

def Gauss_exception_method(A, F, N):
    """Straight Gaussian move"""
    for raw in range(0, N-1):
        ksi = A.item(raw+1, raw) / A.item(raw, raw) # ksi = a_i / b_(i-1)
        b = A.item(raw+1, raw+1)
        c = A.item(raw, raw+1)
        
        A.itemset((raw+1, raw), 0)           # a_i = 0
        A.itemset((raw+1, raw+1), b - ksi*c) # b_i = b_i - ksi*c_(i-1)
        
        F[raw + 1] -= ksi*F[raw]
    
    """Reverse Gaussian move"""
    y = np.zeros(N)
    y[N-1] = F[N-1] / A.item(N-1, N-1)  # y_n = f_n / b_n
    for raw in range(N - 2, -1, -1):
        # y_i = (f_i - c_i * y_(i+1)) / b_i
        y[raw] = (F[raw] - A.item(raw,raw+1) * y[raw+1]) / A.item(raw,raw)
    
    return y

if __name__ == "__main__":
    print("Input number of dotes:")
    N = int(input())
    print("Input edge conditions:")
    y_0 = float(input())

    x = np.linspace(0, np.pi, N)
    h = x[1] - x[0]
    A, F = create_matrix(N, h, y_0)
    y = Gauss_exception_method(A, F, N)
    analitic_y = solv_func(x, y_0)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    axs[0].plot(x, y, 'ro', label = 'numerical solution')
    axs[0].plot(x, analitic_y, '-', label = 'analitic solution')
    axs[0].legend(fontsize = 15,
          facecolor = 'oldlace',    #  цвет области
          edgecolor = 'r',    #  цвет крайней линии
         )
    axs[0].set_xlabel('x', fontsize = 15, color = 'blue')
    axs[0].set_ylabel('y_error', fontsize = 15, color = 'blue')
    axs[0].set_title('Solving of the dif equation', size=30)
    axs[0].grid()
    
    axs[1].plot(x, abs(y - analitic_y), 'ro', label = 'error')
    axs[1].legend(fontsize = 15,
          facecolor = 'oldlace',    #  цвет области
          edgecolor = 'r',    #  цвет крайней линии
         )
    axs[1].set_xlabel('x', fontsize = 15, color = 'blue')
    axs[1].set_ylabel('y_error', fontsize = 15, color = 'blue')
    axs[1].set_title('Solving of the dif equation', size=30)
    axs[1].grid()
    
    plt.show()
