import numpy as np
from scipy.optimize import minimize

# Определение целевой функции
def f(x):
    x1, x2 = x
    return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2

# Определение барьерной функции
def barrier_function(x, mu):
    # Ограничения
    g1 = 8*x[0] - 3*x[1] - 40  # g1(x) <= 0
    g2 = -2*x[0] + x[1] + 3    # g2(x) = 0
    # Барьерные члены
    barrier_term = -mu * (np.log(-g1) + (g2 ** 2))
    return f(x) + barrier_term

# Метод барьерных функций
def barrier_method(initial_point, mu=1.0, beta=0.5, max_iter=100):
    x = np.array(initial_point)
    
    for i in range(max_iter):
        # Минимизация барьерной функции
        result = minimize(barrier_function, x, args=(mu,), bounds=[(None, None), (None, None)])
        x = result.x
        
        # Уменьшение mu
        mu *= beta
        
        # Проверка на сходимость
        if result.success:
            print(f"Итерация {i + 1}: x = {x}, f(x) = {f(x)}, mu = {mu}")
        else:
            print("Оптимизация не удалась")
            break

    return x

# Начальная точка
initial_point = [1.0, 1.0]
optimal_solution = barrier_method(initial_point)

print(f"Оптимальное решение: {optimal_solution}")
print(f"Оптимальное значение: {f(optimal_solution)}")
