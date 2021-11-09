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
    S_2 /= vib
    return S_2


def math_exp_int(exp, disp, test, vib):
    exp_int = test * math.sqrt(disp / vib)
    return [exp - exp_int, exp + exp_int]


def disp_int(vib, disp, alpha):
    chi1 = scipy.stats.chi2.ppf(alpha + (1 - alpha) / 2, df=vib - 1)
    chi2 = scipy.stats.chi2.ppf((1 - alpha) / 2, df=vib - 1)
    d_int = (vib - 1) * disp
    return [d_int / chi1, d_int / chi2]


def correlation(arrayX, arrayY, expX_emp, expY_emp, sizeX, sizeY, dispX, dispY):
    Mx = 0
    My = 0
    Dx = 0
    Dy = 0
    for i in range(sizeX):
        Mx += (arrayX[i] - expX_emp)
        Dx += (arrayX[i] - expX_emp) ** 2
    for i in range(sizeY):
        My += (arrayY[i] - expY_emp)
        Dy += (arrayY[i] - expY_emp) ** 2
    result = (Mx * My) / ((math.sqrt(Dx) * math.sqrt(Dy)))
    return result / (math.sqrt(sizeX) * math.sqrt(sizeY))


def criteria(matrix_t, matrix_emp, size_x, size_y, vib):
    crit_x = 0
    crit_y = 0
    for i in range(size_x):
        crit_x = (sum(matrix_t[i]) - sum(matrix_emp[i])) ** 2 / sum(matrix_emp[i])
    temp_mat = copy.copy(matrix_emp)
    temp_mat = temp_mat.transpose()
    temp_th = copy.copy(matrix_t)
    temp_th = temp_th.transpose()
    for i in range(size_y):
        crit_y = (sum(temp_th[i]) - sum(temp_mat[i])) ** 2 / sum(temp_mat[i])
    return crit_x, crit_y


n = int(input("n: "))
m = int(input("m: "))
rnd = int(input("rnd: "))
# n = 6
# m = 6
# rnd = 250
alpha = 0.01

prob = []
emp_matrix = np.zeros((n, m))
array = np.array(np.random.dirichlet(np.ones(n * m)) * 1)
theory_matrix = np.reshape(array, (n, m))
print("Теоретическая матрица распределения:")
print(theory_matrix)

for i in range(rnd):
    a = random()
    summ = 0
    skip = False
    for j in range(n):
        for k in range(m):
            summ += theory_matrix[j][k]

            if summ >= a:
                emp_matrix[j][k] += 1
                skip = True
                break
            else:
                continue
        if skip:
            break

for i in range(n):
    for j in range(m):
        emp_matrix[i][j] /= rnd

emp_copy = np.reshape(copy.copy(emp_matrix), (1, n * m))[0]
prob = copy.copy(array)
prob_str = []
for i in range(len(array)):
    prob_str.append(str(prob[i]))
print()

print("Эмпирическая матрица распределения:")
print(emp_matrix)
X = []
Y = []
vib_X = []
vib_Y = []
for i in range(n):
    X.append(sum(emp_matrix[i]))
    vib_X.append(X[i] * rnd)

temp_emp = copy.copy(emp_matrix)
temp_emp = temp_emp.transpose()

for i in range(m):
    Y.append(sum(temp_emp[i]) * rnd)
    vib_Y.append(Y[i] * rnd)
X_str = []
Y_str = []
for i in range(n):
    X_str.append(str(X[i]))
for i in range(m):
    Y_str.append(str(Y[i]))

fig, ax = plt.subplots(figsize=(15, 7))
fig1, ax1 = plt.subplots(figsize=(15, 7))
result = ax.bar(X_str, vib_X)
result2 = ax1.bar(Y_str, vib_Y)
plt.show()

arrX = arrayXY(n)
arrY = arrayXY(m)
print('\n')

expX_theory = math_exp_X(arrX, theory_matrix, n, m, rnd)
expY_theory = math_exp_Y(arrY, theory_matrix, n, m, rnd)

expX_emp = math_exp_X(arrX, emp_matrix, n, m, rnd)
expY_emp = math_exp_Y(arrY, emp_matrix, n, m, rnd)

print("Теоретическая точечная оценка M[X]:", expX_theory)
print("Теоретическая точечная оценка M[Y]:", expY_theory)
print()
print("Эмпирическая точечная оценка M[X]:", expX_emp)
print("Эмпирическая точечная оценка M[Y]:", expY_emp)

D_x_theory = disp_dot('X', n, arrX, expX_theory, rnd)
D_y_theory = disp_dot('Y', m, arrY, expY_theory, rnd)
print()
print("Теоретическая точечная оценка D[X]:", D_x_theory)
print("Теоретическая точечная оценка D[Y]:", D_y_theory)

D_x_emp = disp_dot('X', n, arrX, expX_emp, rnd)
D_y_emp = disp_dot('Y', m, arrY, expY_emp, rnd)
print()
print("Эмпирическая точечная оценка D[X]:", D_x_emp)
print("Эмпирическая точечная оценка D[Y]:", D_y_emp)
print()
test_st = student_test(0.01, rnd - 1)

math_expX_int_emp = math_exp_int(expX_emp, D_x_emp, test_st, rnd)
math_expY_int_emp = math_exp_int(expY_emp, D_y_emp, test_st, rnd)

math_expX_int_th = math_exp_int(expX_theory, D_x_theory, test_st, rnd)
math_expY_int_th = math_exp_int(expY_theory, D_y_theory, test_st, rnd)

print("Теоретическая интервальная оценка M[X]:", math_expX_int_th)
print("Теоретическая интервальная оценка M[Y]:", math_expY_int_th)
print()
print("Эмпирическая интервальная оценка M[X]:", math_expX_int_emp)
print("Эмпирическая интервальная оценка M[Y]:", math_expY_int_emp)

disp_intX_th = disp_int(rnd, D_x_theory, 0.01)
disp_intY_th = disp_int(rnd, D_y_theory, 0.01)

disp_intX_emp = disp_int(rnd, D_x_emp, 0.01)
disp_intY_emp = disp_int(rnd, D_y_emp, 0.01)
print()
print("Теоретическая интервальная оценка D[X]:", disp_intX_th)
print("Теоретическая интервальная оценка D[Y]:", disp_intY_th)
print()
print("Эмпирическая интервальная оценка D[X]:", disp_intX_emp)
print("Эмпирическая интервальная оценка D[Y]:", disp_intY_emp)

print("Теоретический коэффицент корреляции:",
      correlation(arrX, arrY, expX_theory, expY_theory, n, m, D_x_theory, D_y_theory))

print("Эмпирический коэффицент корреляции:", correlation(arrX, arrY, expX_emp, expY_emp, n, m, D_x_emp, D_y_emp))

critX, critY = criteria(theory_matrix, emp_matrix, n, m, rnd)

xiX = scipy.stats.chi2.ppf(alpha, df=n - 3)
xiY = scipy.stats.chi2.ppf(alpha, df=m - 3)
checkX = False
checkY = False
print("Проверка гипотезы по критерию согласия Пирсона:")
print()
print("X - Chi^2:", xiX)
print("Критерий Пирсона для X:", critX)
print()
print("Y - Chi^2:", xiY)
print("Критерий Пирсона для Y:", critY)
if xiX >= critX:
    checkX = True
if xiY >= critY:
    checkY = True
print()
print("Проверка гипотезы о соответствии полученных оценок характеристик СВ требуемым:")
print()
print("X -", checkX)
print("Y -", checkY)
