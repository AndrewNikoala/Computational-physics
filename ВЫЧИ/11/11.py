import numpy as np
import matplotlib.pyplot as plt
"""
    Найти уровень энергии(собственное значение гамильтониана) волновую функцию psi(x) основного
    состояния в потенциальной яме U(x), решая конечномерный аналог спектральной задачи для одномерного
    стационарного уравнения Шрёдингера
    
    (-1/2{d^2/dx^2} + U(x) - E)psi(x,t) = 0
    
    Для поиска наименьшего собственного значения H^psi = E_0 * psi трёхдиагональной матрицы
    H использовать метод обратных итераций. Проверить работу программы, сравнив с точным решением для 
    U(x) = 1/2x^2
"""

def fU(x):
    return x**2 / 2

def create_matrix(N, x, h):
    """Creating of tridiagonal matrix and right part of the system"""   
    
    A = np.zeros((N, N))
    a = -1./ (2*h**2)
    c = -1./ (2*h**2)
    col_offset = 0
        
    for col in range(0, N):
        b = 1./ (h**2) + fU(x[col])
        if col == 0:
            A.itemset((0,0), b)
            A.itemset((0,1), c)
        elif col == N-1:
            A.itemset((N-1, N-1), b)
            A.itemset((N-1, N-2), a)
        else:
            A.itemset((col, col_offset), a)
            A.itemset((col, col_offset+1), b)
            A.itemset((col, col_offset+2), c)
            col_offset += 1
    return A

def getMinE(N, H):
    invH = np.linalg.inv(H) 
    
    u = np.zeros((N,1))
    for j in range(0, N):
        u[j][0] = 1
    
    # Here you can change this parameter by another {N}->{M}
    for j in range(0, N):
        u_prev = u
        u = invH.dot(u)     # inv H multiplied by u
    
    E_0 = u_prev[0] / u[0]
    
    return np.squeeze(E_0), u
if __name__ == "__main__":
    print("Input number of steps:")
    N = int(input())
    
    x = np.linspace(-10, 10, N)
    h = x[1] - x[0]

    H = create_matrix(N, x, h)
    
    E_0, psi_0 = getMinE(N, H)
    print('Энергия основного состояния: {:.5f}'.format(E_0))
    
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    print('Волновой вектор основного состояния: \n{}'.format(psi_0))
    
    fig, axs = plt.subplots(figsize=(12, 10))
    axs.plot(x, psi_0, '-')
    axs.set_xlabel('x', fontsize = 15, color = 'blue')
    axs.set_ylabel('psi_0', fontsize = 15, color = 'blue')
    axs.set_title('Solving of the dif equation', size=30)
    axs.grid()
    
    plt.show()
