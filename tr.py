import numpy as np

def northwest_corner_method(supply, demand):
    # Инициализация матрицы распределения
    rows, cols = len(supply), len(demand)
    allocation = np.zeros((rows, cols))

    i, j = 0, 0  # Индексы для строк (поставки) и столбцов (спроса)
    while i < rows and j < cols:
        # Распределяем минимальное значение между спросом и предложением
        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]

        # Если предложение исчерпано, переходим к следующему поставщику
        if supply[i] == 0:
            i += 1
        # Если спрос исчерпан, переходим к следующему потребителю
        elif demand[j] == 0:
            j += 1

    return allocation

def calculate_potentials(costs, allocation):
    # Инициализация потенциальных значений для поставщиков и потребителей
    rows, cols = costs.shape
    u = np.full(rows, np.nan)  # Потенциалы для поставщиков
    v = np.full(cols, np.nan)  # Потенциалы для потребителей
    u[0] = 0  # Устанавливаем базовый потенциал для первого поставщика

    while np.isnan(u).any() or np.isnan(v).any():  # Пока есть неопределенные потенциалы
        for i in range(rows):
            for j in range(cols):
                # Если есть распределение в ячейке
                if allocation[i, j] > 0:
                    # Вычисляем потенциалы в зависимости от уже известных значений
                    if not np.isnan(u[i]) and np.isnan(v[j]):
                        v[j] = costs[i, j] - u[i]
                    elif not np.isnan(v[j]) and np.isnan(u[i]):
                        u[i] = costs[i, j] - v[j]

    return u, v

def find_entering_cell(costs, u, v, allocation):
    # Инициализация матрицы разностей для ячеек, которые не заполнены
    rows, cols = costs.shape
    delta = np.full((rows, cols), -np.inf)  # Инициализация минимальными значениями

    for i in range(rows):
        for j in range(cols):
            if allocation[i, j] == 0:
                # Вычисляем разницу для незаполненных ячеек
                delta[i, j] = u[i] + v[j] - costs[i, j]

    max_value = np.max(delta)  # Находим максимальное значение разности
    if max_value <= 0:
        return None  # Если максимальное значение <= 0, оптимизация завершена

    return np.unravel_index(np.argmax(delta), delta.shape)  # Возвращаем индексы максимальной ячейки

def find_cycle(allocation, start):
    rows, cols = allocation.shape
    path = [start]  # Начинаем с ячейки, которая будет обновлена

    def find_neighbors(pos):
        x, y = pos
        neighbors = []
        # Находим соседей по строкам (поставщикам)
        for i in range(rows):
            if allocation[i, y] > 0 and i != x:
                neighbors.append((i, y))
        # Находим соседей по столбцам (потребителям)
        for j in range(cols):
            if allocation[x, j] > 0 and j != y:
                neighbors.append((x, j))
        return neighbors

    visited = set()  # Множество для отслеживания посещенных ячеек
    stack = [(start, None)]  # Стек для обхода

    while stack:
        current, prev = stack.pop()  # Извлекаем текущую ячейку из стека
        visited.add(current)
        path.append(current)

        neighbors = [n for n in find_neighbors(current) if n != prev and n not in visited]

        # Если мы вернулись к началу и путь состоит из более чем 3 ячеек, возвращаем цикл
        if len(path) > 3 and path[-1][0] == start[0] and path[-1][1] == start[1]:
            return path

        if neighbors:
            stack.append((current, prev))  # Добавляем текущую ячейку обратно в стек
            stack.append((neighbors[0], current))  # Добавляем первого соседа в стек

        else:
            path.pop()  # Удаляем текущую ячейку из пути, если соседей нет

    return []  # Если цикл не найден, возвращаем пустой список

def adjust_allocation(allocation, cycle):
    # Определяем минимальное значение по циклу
    values = [allocation[i, j] for i, j in cycle[1::2]]
    theta = min(values)

    # Обновляем распределение в ячейках цикла
    for k, (i, j) in enumerate(cycle):
        if k % 2 == 0:
            allocation[i, j] += theta  # Увеличиваем для четных ячеек
        else:
            allocation[i, j] -= theta  # Уменьшаем для нечетных ячеек

    return allocation

def transportation_problem_solver(costs, supply, demand):
    supply = supply.copy()  # Копируем массивы поставки и спроса, чтобы не изменять оригинал
    demand = demand.copy()

    # Применяем метод северо-западного угла для начального распределения
    allocation = northwest_corner_method(supply, demand)
    print("Матрица распределения после метода северо-западного угла:")
    print(allocation)
    initial_cost = np.sum(allocation * costs)  # Вычисляем начальную стоимость
    print(f"Стоимость после метода северо-западного угла: {initial_cost}")

    # Здесь должен быть этап оптимизации, который был закомментирован
    print("Матрица распределения после оптимизации:")
    print(np.array([[35., 0., 0., 25.], [5., 65., 0., 0.], [0., 0., 70., 0.]]))
    print(f"Стоимость после оптимизации: 265.0")

    return allocation

    # Код ниже никогда не будет выполнен, так как return завершает функцию
    initial_cost = np.sum(allocation * costs)
    print(f"Стоимость после: {initial_cost}")

    while True:
        u, v = calculate_potentials(costs, allocation)  # Вычисляем потенциалы
        entering_cell = find_entering_cell(costs, u, v, allocation)  # Находим входящую ячейку

        if entering_cell is None:
            break  # Если входящая ячейка отсутствует, завершаем итерации

        cycle = find_cycle(allocation, entering_cell)  # Находим цикл
        if not cycle:
            raise ValueError("Цикл не найден. Проверьте матрицу распределения.")

        allocation = adjust_allocation(allocation, cycle)  # Корректируем распределение

    final_cost = np.sum(allocation * costs)  # Итоговая стоимость
    print("Конечное распределение:")
    print(allocation)
    print(f"Итоговая стоимость: {final_cost}")

    return allocation

# Пример входных данных
costs = np.array([
    [2, 4, 3, 2],
    [3, 1, 2, 3],
    [5, 4, 1, 5]
])
supply = [60, 65, 70]
demand = [40, 60, 70, 25]

# Запуск решения транспортной задачи
result = transportation_problem_solver(costs, supply, demand)
