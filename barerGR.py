import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def f(x):
    x1, x2 = x[0], x[1]
    return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2

# Определяем градиент функции
def gradient(x):
    x1, x2 = x
    df_dx1 = 2*x1 - 3*x2 + 5  # Частная производная по x1
    df_dx2 = -3*x1 + 20*x2 - 3  # Частная производная по x2
    return np.array([df_dx1, df_dx2])

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
    return f(x) + barrier_term

# Градиентный метод с барьерной функцией
def barrier_gradient_descent(initial_point, initial_step=0.1, tol=1e-6, max_iter=1000, mu=1.0):
    x = np.array(initial_point)
    step_size = initial_step
    iter_count = 0

    while iter_count < max_iter:
        # Вычисляем градиент барьерной функции
        grad = gradient(x)  # Градиент целевой функции
        barrier_grad = np.array([2*x[0] - 3*x[1] + 5, -3*x[0] + 20*x[1] - 3])  # Градиент целевой функции

        # Добавляем градиенты штрафной функции
        if 8 * x[0] - 3 * x[1] - 40 > 0:  # Если g1 нарушено
            barrier_grad[0] += -mu * (8)  # Добавляем штраф для g1
        if -2 * x[0] + x[1] + 3 != 0:  # Если g2 нарушено
            barrier_grad[1] += -mu * (-2)  # Добавляем штраф для g2

        total_grad = grad + barrier_grad  # Суммируем градиенты

        # Оптимизация шага (линейный поиск)
        step_size = 1.0  # Начальный шаг
        while barrier_function(x - step_size * total_grad, mu) >= barrier_function(x, mu):
            step_size *= 0.5  # Уменьшаем шаг, пока не получим улучшение

        # Обновляем значение
        x_new = x - step_size * total_grad  # Обновляем значение с учетом градиента барьерной функции
        
        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new  # Обновляем x

        # Уменьшение mu
        mu *= 0.5
        iter_count += 1

    return x, f(x), iter_count  # Возвращаем координаты минимума, значение функции и количество итераций

# Начальная точка
x0 = np.array([2.0, 1.0])

# Запуск градиентного спуска с барьерной функцией
min_point, min_value, iterations = barrier_gradient_descent(x0)

# Вывод результатов
print(f"Координаты точки минимума: {min_point}")
print(f"Минимальное значение функции: {min_value}")
print(f"Количество итераций: {iterations}")

# Построение графика функции
x1 = np.linspace(-2, 4, 400)
x2 = np.linspace(-2, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(np.array([X1, X2]))

# Создание графика
plt.figure(figsize=(10, 6))
contour = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.plot(min_point[0], min_point[1], 'ro')  # Точка минимума
plt.title("Контурная карта функции")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.show()
