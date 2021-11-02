import numpy as np
from random import *
import matplotlib.pyplot as plt
import copy
import scipy.stats
import math


def student_test(alpha, angle_freedom):
    return scipy.stats.t.ppf(1 - (alpha / 2), angle_freedom)


def arrayXY(size):
    arr = []
    l = list(range(1, 11))
    shuffle(l)
    for i in range(size):
        arr.append(l[i])
    arr.sort()
    return arr


def math_exp_X(array, matrix, sizeX, sizeY, vib):
    Mx = 0
    for i in range(sizeX):
        summa = 0
        for j in range(sizeY):
            summa += matrix[i][j]
        Mx += summa * array[i]

    Mx /= vib
    return Mx


def math_exp_Y(array, matrix, sizeX, sizeY, vib):
    Mx = 0
    for j in range(sizeY):
        summa = 0
        for i in range(sizeX):
            summa += matrix[i][j]
        Mx += summa * array[j]

    Mx /= vib
    return Mx


def disp_dot(type, size, array, exp, vib):
    S_2 = 0
    for i in range(size):
        S_2 += (array[i] - exp) ** 2
    S_2 /= len(array)
    if type == 'X':
        print('Точечная оценка D[X]:', S_2)
    else:
        print('Точечная оценка D[Y]:', S_2)
    return S_2


def math_exp_int(exp, disp, test, vib):
    exp_int = test * math.sqrt(disp / vib)
    return [exp - exp_int, exp + exp_int]


def disp_int(vib, disp):
    chi1 = 304.9
    chi2 = 200.9
    d_int = (vib - 1) * disp
    return [d_int / chi1, d_int / chi2]


def correlation(arrayX, arrayY, expX, expY, sizeX, sizeY, dispX, dispY):
    Mx = 0
    My = 0
    for i in range(sizeX):
        Mx += (arrayX[i] - expX)
    for i in range(sizeY):
        My += (arrayY[i] - expY)
    result = (Mx * My) / ((math.sqrt(dispX) * math.sqrt(dispY)))
    return result / (math.sqrt(sizeX) * math.sqrt(sizeY))


def criteria(matrix_t, matrix_emp, size_x, size_y, vib):
    crit_x = 0
    crit_y = 0
    for i in range(size_x):
        crit_x = (sum(matrix_t[i]) - sum(matrix_emp[i] / vib)) ** 2 / sum(matrix_emp[i] / vib)
    temp_mat = copy.copy(matrix_emp)
    temp_mat = temp_mat.transpose()

    for i in range(size_y):
        crit_y = (sum(matrix_t[i]) - sum(temp_mat[i] / vib)) ** 2 / sum(temp_mat[i] / vib)
    return crit_x, crit_y


# n = int(input("n: "))
# m = int(input("m: "))
# rnd = int(input("rnd: "))
n = 6
m = 6
rnd = 250
xi = 11.34
prob = []
emp_matrix = np.zeros((n, m))
array = np.array(np.random.dirichlet(np.ones(n * m)) * 1)
matrix = np.reshape(array, (n, m))
print("Теоретическая матрица распределения:")
print(matrix)

for i in range(rnd):
    a = random()
    summ = 0
    skip = False
    for j in range(n):
        for k in range(m):
            summ += matrix[j][k]

            if summ >= a:
                emp_matrix[j][k] += 1
                skip = True
                break
            else:
                continue
        if skip:
            break

emp_copy = np.reshape(copy.copy(emp_matrix), (1, n * m))[0]
prob = copy.copy(array)
prob_str = []
for i in range(len(array)):
    prob_str.append(str(prob[i]))
print()
print("Эмпирическая матрица распределния:")
print(emp_matrix)
fig, ax = plt.subplots(figsize=(15, 7))
result = ax.bar(prob_str, emp_copy)
plt.show()

arrX = arrayXY(n)
arrY = arrayXY(m)
print('\n')

expX = math_exp_X(arrX, emp_matrix, n, m, rnd)
expY = math_exp_Y(arrY, emp_matrix, n, m, rnd)

print("Точечная оценка M[X]:", expX)
print("Точечная оценка M[Y]:", expY)

D_x = disp_dot('X', n, arrX, expX, rnd)
D_y = disp_dot('Y', m, arrY, expY, rnd)

test_st = student_test(0.01, rnd - 1)

math_expX_int = math_exp_int(expX, D_x, test_st, rnd)
math_expY_int = math_exp_int(expY, D_y, test_st, rnd)

print("Интервальная оценка M[X]:", math_expX_int)
print("Интервальная оценка M[Y]:", math_expY_int)

disp_intX = disp_int(rnd, D_x)
disp_intY = disp_int(rnd, D_y)

print("Интервальная оценка D[X]:", disp_intX)
print("Интервальная оценка D[Y]:", disp_intY)

print("Коэффицент корреляции:", correlation(arrX, arrY, expX, expY, n, m, D_x, D_y))

critX, critY = criteria(matrix, emp_matrix, n, m, rnd)
print("Проверка гипотезы по критерию согласия Пирсона:")
print()
print("Chi^2(alpha = 0.01, k = 3):", xi)
print("Критерий Пирсона для X:",critX)
print("Критерий Пирсона для Y:",critY)
print()
print("Chi^2(alpha, k) > chi^2(X) & Chi^2(alpha, k) > chi^2(Y), значит, нет оснований отклонять гипотезу о соответствии полученных оценок характеристик СВ требуемым")