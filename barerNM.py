import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def f(x1, x2):
    return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2

# Определяем барьерную функцию
def barrier_function(x, mu):
    x1, x2 = x
    # Ограничения
    g1 = 8 * x1 - 3 * x2 - 40  # g1(x) <= 0
    g2 = -2 * x1 + x2 + 3      # g2(x) = 0
    # Проверка на допустимость
    if g1 >= 0 or g2 != 0:
        return np.inf  # Если ограничения нарушены, возвращаем бесконечность
    # Барьерные члены
    barrier_term = -mu * (np.log(-g1) + (g2 ** 2))
    return f(x1, x2) + barrier_term

# Метод Нелдера-Мида
def nelder_mead(func, x0, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-6, max_iter=1000):
    n = len(x0)  # Размерность (количество переменных)
    simplex = np.array([x0, x0 + np.array([alpha, 0]), x0 + np.array([0, alpha])])  # Начальный симплекс
    values = np.array([func(simplex[i]) for i in range(n + 1)])  # Значения функции в вершинах симплекса

    for iteration in range(max_iter):
        # Сортируем вершины по значениям функции
        order = values.argsort()
        simplex = simplex[order]
        values = values[order]

        # Центрирование
        centroid = np.mean(simplex[:-1], axis=0)

        # Отражение
        reflected = centroid + alpha * (centroid - simplex[-1])
        reflected_value = func(reflected)

        if values[0] <= reflected_value < values[-2]:
            simplex[-1] = reflected
            values[-1] = reflected_value
            continue

        # Увеличение
        if reflected_value < values[0]:
            expanded = centroid + gamma * (centroid - simplex[-1])
            expanded_value = func(expanded)
            if expanded_value < reflected_value:
                simplex[-1] = expanded
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected
                values[-1] = reflected_value
            continue

        # Сжатие
        contracted = centroid + beta * (simplex[-1] - centroid)
        contracted_value = func(contracted)

        if contracted_value < values[-1]:
            simplex[-1] = contracted
            values[-1] = contracted_value
        else:
            # Уменьшаем симплекс
            simplex = simplex[0] + (simplex - simplex[0]) * beta
            values = np.array([func(simplex[i]) for i in range(n + 1)])

        # Проверка на сходимость
        if np.max(np.abs(values - values[0])) < tol:
            break

    return simplex[0], values[0], iteration + 1  # Возвращаем координаты минимума, значение функции и количество итераций

# Метод барьерных функций
def barrier_method(initial_point, mu=1.0, beta=0.5, max_iter=100):
    x = np.array(initial_point)
    
    for i in range(max_iter):
        # Минимизация барьерной функции с помощью метода Нелдера-Мида
        result = nelder_mead(lambda x: barrier_function(x, mu), x)
        x = result[0]  # Обновляем x на найденную точку

        # Уменьшение mu
        mu *= beta
        
        # Проверка на сходимость
        if np.abs(barrier_function(x, mu) - barrier_function(result[0], mu)) < 1e-6:
            break

    return x, barrier_function(x, mu)

# Начальная точка
x0 = np.array([2, 1])

# Запуск метода барьерных функций
optimal_solution, optimal_value = barrier_method(x0)

# Вывод результатов
print(f"Координаты точки минимума: {optimal_solution}")
print(f"Минимальное значение функции: {optimal_value}")

# Построение графика функции
x1 = np.linspace(-2, 4, 400)
x2 = np.linspace(-2, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Создание графика
plt.figure(figsize=(10, 6))
contour = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.plot(optimal_solution[0], optimal_solution[1], 'ro')  # Точка минимума
plt.title("Контурная карта функции")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.show()
