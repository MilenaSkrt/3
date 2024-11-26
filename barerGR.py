import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def f(x1, x2):
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
    return f(x1, x2) + barrier_term

# Градиентный метод с барьерной функцией
def barrier_gradient_descent(initial_point, mu=1.0, beta=0.5, initial_step=0.1, tol=1e-6, max_iter=1000):
    x = np.array(initial_point)
    step_size = initial_step
    
    for iteration in range(max_iter):
        # Вычисляем градиент барьерной функции
        grad = gradient(x)  # Градиент целевой функции
        barrier_grad = np.array([2*x[0] - 3*x[1] + 5, -3*x[0] + 20*x[1] - 3])  # Градиент барьерной функции

        # Обновляем значение
        x_new = x - step_size * (grad + barrier_grad)  # Обновляем значение с учетом градиента барьерной функции
        
        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            break
        
        # Проверка, улучшилось ли значение функции
        if barrier_function(x_new, mu) < barrier_function(x, mu):
            x = x_new  # Если улучшилось, обновляем x
        else:
            step_size *= 0.5  # Если не улучшилось, уменьшаем шаг

        # Уменьшение mu
        mu *= beta

    return x, f(*x), iteration + 1  # Возвращаем координаты минимума, значение функции и количество итераций

# Начальная точка
x0 = np.array([2, 1])

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
Z = f(X1, X2)

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
