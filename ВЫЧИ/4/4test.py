import math

N = 100
eps = 1e-10

a = 0
b = math.pi

def func(x, m, t):
    return math.cos(m * t - x * math.sin(t)) / math.pi  # (0, pi)
# calculating of Bessel`s Integral
def integrate(x, m):
    delta_t = (b - a) / N
    summa = 0
    for i in range(N):
        a_i = a + i * delta_t
        b_i = a + (i + 1) * delta_t
        summa = summa + (func(x, m, a_i) + 4 * func(x, m, (a_i + b_i) / 2) + func(x, m, b_i)) * delta_t / 6
    return summa
# derivative of Bessel`s function
def derivative(x, m):
    dx = 0.00001
    return (integrate(x + dx, m) - integrate(x - dx, m)) / (2 * dx)


if __name__ == '__main__':
    n = 100
    x_0 =  2 * math.pi / n
    for i in range(0, n):
        x = x_0 * i 
        res = derivative(x, 0) + integrate(x, 1)
        res1 = derivative(x, 0)
        res2 = integrate(x, 1)
        print("%1.10f" % res1, " | ", "%1.10f" %res2)
        if res > eps:
            print()
            print("precision not enough", i)
            break
