import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def f(x):
    x1, x2 = x[0], x[1]
    return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2

# Функция штрафа
def penalty(x):
    g = x[0] + 8 * x[1] + 5  # Уравнение плоскости
    return g**2 if g > 0 else 0  # Штраф только если ограничение нарушено

# Новая целевая функция с штрафом
def penalized_function(x, penalty_weight=1000):
    return f(x) + penalty_weight * penalty(x)

# Метод Нелдера-Мида (Метод деформируемого многогранника)
def nelder_mead(f, x0, tol=1e-6, max_iter=1000):
    # Коэффициенты для алгоритма
    a = 1.0  # коэффициент отражения
    y = 2.0  # коэффициент растяжения
    b = 0.5  # коэффициент сжатия
    sigma = 0.5  # коэффициент уменьшения

    # Создание начального симплекса вокруг точки x0
    n = len(x0)
    simplex = [x0]

    # Смещение для создания симплекса
    shift = 0.05  # Выбираем смещение на 0.05 для начала

    # Смещение для создания соседних точек
    for i in range(n):
        x = np.copy(x0)
        x[i] += shift  # Смещаем по оси i
        simplex.append(x)

    # Вычисляем значения функции в вершинах симплекса
    f_values = [f(x) for x in simplex]

    iter_count = 0  # Счетчик итераций
    simplex_history = [np.array(simplex)]  # История симплексов для графика

    # Основной цикл оптимизации
    while iter_count < max_iter:
        # Сортировка вершин симплекса по значению функции
        indices = np.argsort(f_values)
        simplex = [simplex[i] for i in indices]
        f_values = [f_values[i] for i in indices]

        # Вычисление центра тяжести (исключая наихудшую точку)
        centroid = np.mean(simplex[:-1], axis=0)

        # Отражение
        xr = centroid + a * (centroid - simplex[-1])
        fr = f(xr)

        # Проверка условий для отражения, растяжения и сжатия
        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        elif fr < f_values[0]:
            # Растяжение, если отраженная точка лучше
            xe = centroid + y * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        else:
            # Сжатие
            xc = centroid + b * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Уменьшение симплекса
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                f_values = [f(x) for x in simplex]

        # Сохранение текущего симплекса для визуализации
        simplex_history.append(np.array(simplex))

        # Условие остановки
        if np.max(np.abs(np.array(f_values) - f_values[0])) < tol:
            break

        iter_count += 1

    return simplex[0], f_values[0], iter_count, simplex_history


# Начальная точка
x0 = np.array([1.0, 1.0])

# Запуск метода Нелдера-Мида с учетом штрафа
penalty_weight = 1000  # Коэффициент штрафа
minimum_penalty, f_min_penalty, iterations_penalty, simplex_history_penalty = nelder_mead(
    lambda x: penalized_function(x, penalty_weight), x0
)

print("Минимум функции с учетом штрафа:", minimum_penalty)
print("Значение функции в минимуме (с учетом штрафа):", f_min_penalty)
print("Количество итераций:", iterations_penalty)

# Построение графика
x_range = np.linspace(-2, 4, 200)
y_range = np.linspace(-2, 4, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = f(np.array([X, Y]))

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="f(x)")

# Отображение симплексов
for simplex in simplex_history_penalty:
    plt.plot(simplex[:, 0], simplex[:, 1], 'k-', alpha=0.4)
    plt.plot(simplex[:, 0], simplex[:, 1], 'ko', markersize=3)

# Отображение ограничения
x_line = np.linspace(-2, 4, 200)
y_line = -(x_line + 5) / 8
plt.plot(x_line, y_line, 'r--', label='Ограничение: $x_1 + 8x_2 + 5 = 0$')

# Точки минимума
plt.plot(x0[0], x0[1], 'bo', label='Начальная точка (x0)')
plt.plot(minimum_penalty[0], minimum_penalty[1], 'ro', label='Минимум функции с учетом штрафа')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Минимизация функции методом Нелдера-Мида с учетом штрафа")
plt.legend()

plt.show()
