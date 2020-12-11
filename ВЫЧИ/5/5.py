import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def make_dots(n, x_i, y_i):
    for k in range(0, n):
        x_i[k] = 1 + k / n
        y_i[k] = math.log(x_i[k])
  
def l_i(x, x_i, i):
    l = 1
    for j in range(0, len(x_i)):
        if(j != i):
            l = l * (x - x_i[j])
    return l
  
def lagrange_polynomial(x, x_i, y_i):
    pol = 0
    for i in range(0, len(x_i)):
      pol = pol + y_i[i] * l_i(x, x_i, i) / l_i(x_i[i], x_i, i)
    return pol


if __name__ == '__main__':
    print("Input x from (1, 2):")
    x = float(input())
    for n in range(399, 400):
        x_i = np.zeros(n)
        y_i = np.zeros(n)
        make_dots(n, x_i, y_i)
        pol = lagrange_polynomial(x, x_i, y_i)

        #building of graphs
        pol_arr = np.zeros(n)
        analitic_func = np.zeros(n)
        delta_f = np.zeros(n)
        
        
        for i in range(0, n):
            analitic_func[i] = math.log(x_i[i])
            pol_arr[i] = lagrange_polynomial(x_i[i], x_i, y_i)
            delta_f[i] = abs(pol_arr[i] - analitic_func[i])
    
fig, ax = plt.subplots()

#ax.plot(x_i, pol_arr, 'ro', x_i, analitic_func, '-')
ax.plot(x_i, delta_f)
ax.set(xlabel='x', ylabel='delta_f',
    title='Interpolation')
ax.grid()
plt.show()
