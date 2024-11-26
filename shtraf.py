# cook your dish here
import numpy as np
from scipy.optimize import minimize

# Определение целевой функции
def f(x):
    x1, x2 = x
    return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2

# Определение штрафной функции
def penalty_function(x, penalty):
    x1, x2 = x
    # Ограничения
    g1 = 8*x1 - 3*x2 + 3  # g1(x) <= 0
    g2 = -2*x1 + x2 - 3   # g2(x) = 0
    # Штраф за нарушения ограничений
    penalty_term = penalty * (max(0, g1) + (g2 ** 2))  # Штраф за неравенство и равенство
    return f(x) + penalty_term

# Метод штрафных функций
def penalty_method(initial_point, penalty=1.0, beta=10, max_iter=100):
    x = np.array(initial_point)
    
    for i in range(max_iter):
        # Минимизация штрафной функции
        result = minimize(penalty_function, x, args=(penalty,), bounds=[(None, None), (None, None)])
        x = result.x
        
        # Увеличение штрафного параметра
        penalty *= beta
        
        # Проверка на сходимость
        if result.success:
            print(f"Итерация {i + 1}: x = {x}, f(x) = {f(x)}, penalty = {penalty}")
        else:
            print("Оптимизация не удалась")
            break

    return x

# Начальная точка
initial_point = [1.0, 1.0]
optimal_solution = penalty_method(initial_point)

print(f"Оптимальное решение: {optimal_solution}")
print(f"Оптимальное значение: {f(optimal_solution)}")
