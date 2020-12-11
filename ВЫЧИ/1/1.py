import sys
import math
e = sys.float_info.epsilon
print('epsilon  =  ',e)
BitnessMantissa = sys.float_info.mant_dig
print ('Bitness of Mantissa  =  ',BitnessMantissa)
min_exp = sys.float_info.min_exp
print ('min exp  =  ', min_exp)
max_exp = sys.float_info.max_exp
print ('max exp  =  ', max_exp)
print()

#Comparison
a = 1
b = 1 + e/2
c = 1 + e
d = 1 + e + e/2
print('a  =  1            =  ', a)
print('b  =  1 + e/2      =  ', b)
print('c  =  1 + e        =  ', c)
print('d  =  1 + e + e/2  =  ', d)

print('\n_____________________________\n')


#-------------------
# Search of a machine epsilon
e = 1
one = 1
while (one + e) != one:
    e = e / 2
e = e*2
print('epsilon  =  ', e)
print()

# Search of a digit capacity of mantissa
rzrd_mantissa = int(abs(math.log2(e)))
print('Bitness of Mantissa  =  ', rzrd_mantissa)
print()

# Search of max exp and min exp
m = 1.
max_num = 0
i = 0
#while math.isfinite(m):
while m - max_num > e:
    max_num = m
    m = m * 2
    i = i + 1
print("max num = ", max_num, "| rzrd_max = ", i - 1)
print("max exp = ", i - 1)
print()

m = 1.
min_num = 1
i = 0
#while m != 0:
while m != 0:
    min_num = m
    m = m / 2
    i = i + 1
print("min num = ", min_num, "| rzrd_min = ", i - 1)
print("min exp = ", -(i - rzrd_mantissa - 1))
print()


# Comparison

a = 1
b = 1 + e/2
c = 1 + e
d = 1 + e + e/2
f = 1 + e/2 + e
print('a  =  1            =  ', a)
print('b  =  1 + e/2      =  ', b)
print('c  =  1 + e        =  ', c)
print('d  =  1 + e + e/2  =   %.16f' % d)
print('f  =  1 + e/2 + e  =   %.16f' % f)
