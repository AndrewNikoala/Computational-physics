import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sum1tr = []
sum1simp = []
sum2tr = []
sum2simp = []
numbers = []

def func_1(x):
    return 1. / (1 + x ** 2)  # (-1, 1)

def func_2(x):
    return x ** (1. / 3) * math.exp(math.sin(x))  # (0, 1)

N = 2

#print("Input precision from 1 to 100:")
#i = int(input())

#N = RATE * i
#print("N = ", N)

for i in range(1, 7):
    N = N * 2
    numbers.append(N)
    print("precision is: ", N)
    # Integral 1
    print("Solution of the first integral:")
    a = -0.000001
    b = 0.000001
    delta_x = (b - a) / N

    # Trapeze Method
    summa = 0
    for i in range(N):
        a_i = a + i * delta_x
        b_i = a + (i + 1) * delta_x
        summa = summa + (func_1(a_i) + func_1(b_i)) * delta_x / 2
    sum1tr.append(summa)
    print("Trapeze Method: sum = ", summa)


    # Simpson`s Method
    summa = 0
    for i in range(N):
        a_i = a + i * delta_x
        b_i = a + (i + 1) * delta_x
        summa = summa + (func_1(a_i) + 4 * func_1((a_i + b_i) / 2) + func_1(b_i)) * delta_x / 6
    sum1simp.append(summa)
    print("Simpson`s Method: sum = ", summa)
    print()

    #Integral 2
    print("Solution of the second integral:")
    a = 0
    b = 1
    delta_x = (b - a) / N

    # Trapeze Method
    summa = 0
    for i in range(N):
        a_i = a + i * delta_x
        b_i = a + (i + 1) * delta_x
        summa = summa + (func_2(a_i) + func_2(b_i)) * delta_x / 2
    sum2tr.append(summa)
    print("Trapeze Method: sum = ", summa)

    # Simpson`s Method
    summa = 0
    for i in range(N):
        a_i = a + i * delta_x
        b_i = a + (i + 1) * delta_x
        summa = summa + (func_2(a_i) + 4 * func_2((a_i + b_i) / 2) + func_2(b_i)) * delta_x / 6
    sum2simp.append(summa)
    print("Simpson`s Method: sum = ", summa)
    print("\n")

# Graphics

# Trapeze Method Integral 1
fig, ax = plt.subplots()
ax.plot(numbers, sum1tr)
ax.set(xlabel='N', ylabel='summa',
    title='')
ax.grid()
#fig.savefig("TrMethInt1.png")
plt.show()

# Simpson`s Method Integral 1
fig, ax = plt.subplots()
ax.plot(numbers, sum1simp)
ax.set(xlabel='N', ylabel='summa',
    title='Simpson`s Method Integral 1')
ax.grid()
#fig.savefig("SimpMethInt1.png")
plt.show()

# Trapeze Method Integral 2
fig, ax = plt.subplots()
ax.plot(numbers, sum2tr)
ax.set(xlabel='N', ylabel='summa',
    title='')
ax.grid()
#fig.savefig("TrMethInt2.png")
plt.show()

# Simpson`s Method Integral 2
fig, ax = plt.subplots()
ax.plot(numbers, sum1simp)
ax.set(xlabel='N', ylabel='summa',
    title='Simpson`s Method Integral 2')
ax.grid()
#fig.savefig("SimpMethInt2.png")
plt.show()
