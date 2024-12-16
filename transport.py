class Cell:
    def __init__(self, row, col):
        self.row = row  # Индекс строки (поставщик)
        self.col = col  # Индекс столбца (потребитель)


class State:
    def __init__(self, cell, prev_dir, next_cells):
        self.cell = cell  # Текущая ячейка
        self.prev_dir = prev_dir  # Направление предыдущего шага
        self.next_cells = next_cells  # Следующие ячейки для исследования


def solve(a, b, costs):
    # Проверка соответствия размеров
    assert len(costs) == len(a)
    assert len(costs[0]) == len(b)

    # Суммы запасов и потребностей
    a_sum = sum(a)
    b_sum = sum(b)

    # Уравнивание запасов и потребностей
    if a_sum > b_sum:
        b.append(a_sum - b_sum)  # Добавляем "сверхпотребность"
        for row in costs:
            row.append(0)  # Добавляем нулевую стоимость
    elif a_sum < b_sum:
        a.append(b_sum - a_sum)  # Добавляем "сверхзапас"
        costs.append([0] * len(b))  # Добавляем нулевую строку в матрицу затрат

    # Инициализация матрицы перевозок
    x = [[0] * len(b) for _ in range(len(a))]
    a_copy = a[:]  # Копируем запасы
    b_copy = b[:]  # Копируем потребности

    indexes_for_baza = []  # Список базисных ячеек
    i, j = 0, 0  # Индексы для прохода по запасам и потребностям
    while True:
        if a_copy[i] < b_copy[j]:
            x[i][j] = a_copy[i]  # Заполняем ячейку
            indexes_for_baza.append(Cell(i, j))  # Добавляем ячейку в базисные
            b_copy[j] -= a_copy[i]  # Уменьшаем потребность
            a_copy[i] = 0  # Обнуляем запас
            i += 1  # Переходим к следующему поставщику
        else:
            x[i][j] = b_copy[j]  # Заполняем ячейку
            indexes_for_baza.append(Cell(i, j))  # Добавляем ячейку в базисные
            a_copy[i] -= b_copy[j]  # Уменьшаем запас
            b_copy[j] = 0  # Обнуляем потребность
            j += 1  # Переходим к следующему потребителю

        # Проверка на завершение
        if sum(a_copy) == 0 and sum(b_copy) == 0:
            print("Метод северо-западного угла завершен")
            break

    # Вычисляем итоговые затраты
    result = sum(x[cell.row][cell.col] * costs[cell.row][cell.col] for cell in indexes_for_baza)
    print(f"Z = {result} (метод северо-западного угла)")

    # Переход к методу потенциалов
    potential_method(a, b, x, costs, indexes_for_baza)


def potential_method(a, b, x, costs, indexes_for_baza):
    m, n = len(a), len(b)  # Количество поставщиков и потребителей

    while True:
        # Сортируем базисные ячейки по строкам и столбцам
        indexes_for_baza.sort(key=lambda cell: (cell.row, cell.col))

        # Массивы для хранения потенциалов
        u = [0] * m  # Потенциалы поставщиков
        v = [0] * n  # Потенциалы потребителей
        fill_u = [False] * m  # Заполнение потенциалов поставщиков
        fill_v = [False] * n  # Заполнение потенциалов потребителей
        fill_u[0] = True  # Устанавливаем начальное значение для первого поставщика

        # Вычисляем потенциалы
        while not all(fill_u) or not all(fill_v):
            for cell in indexes_for_baza:
                i, j = cell.row, cell.col  # Получаем индексы ячейки
                if fill_u[i]:  # Если потенциал поставщика заполнен
                    v[j] = costs[i][j] - u[i]  # Вычисляем потенциал потребителя
                    fill_v[j] = True  # Помечаем, что потенциал потребителя заполнен
                elif fill_v[j]:  # Если потенциал потребителя заполнен
                    u[i] = costs[i][j] - v[j]  # Вычисляем потенциал поставщика
                    fill_u[i] = True  # Помечаем, что потенциал поставщика заполнен

        not_optimal_cells = []  # Список не оптимальных ячеек
        economies = []  # Список экономий
        for i in range(m):
            for j in range(n):
                if all(cell.row != i or cell.col != j for cell in
                       indexes_for_baza):  # Проверяем, является ли ячейка не базисной
                    diff = u[i] + v[j] - costs[i][j]  # Вычисляем разницу
                    if diff > 0:  # Если разница положительная, ячейка не оптимальна
                        not_optimal_cells.append(Cell(i, j))  # Добавляем не оптимальную ячейку
                        economies.append(diff)  # Сохраняем экономию

        if not not_optimal_cells:  # Если нет не оптимальных ячеек
            print("Метод потенциалов завершен")
            print(f"ui = {u}")  # Выводим потенциалы поставщиков
            print(f"vi = {v}")  # Выводим потенциалы потребителей
            break

        # Обработка ячейки с максимальной экономией
        max_economy = max(economies)
        cells_with_max_economy = [cell for cell, economy in zip(not_optimal_cells, economies) if economy == max_economy]

        # Находим ячейку с минимальной стоимостью среди ячеек с максимальной экономией
        min_cost_cell = min(cells_with_max_economy, key=lambda cell: costs[cell.row][cell.col])
        indexes_for_baza.append(min_cost_cell)  # Добавляем ячейку в базисные

        # Строим путь от новой ячейки до базисных
        path = build_path(min_cost_cell, indexes_for_baza)

        # Ячейки, которые будут уменьшены
        minus_cells = path[1::2]
        min_x_value = min(x[cell.row][cell.col] for cell in minus_cells)  # Находим минимальное значение из этих ячеек

        # Обновляем матрицу перевозок
        for idx, cell in enumerate(path):
            if idx % 2 == 0:  # Если индекс четный, увеличиваем значение
                x[cell.row][cell.col] += min_x_value
            else:  # Если индекс нечетный, уменьшаем значение
                x[cell.row][cell.col] -= min_x_value

        # Удаляем ячейки, которые стали нулевыми из базисных ячеек
        for cell in minus_cells:
            if x[cell.row][cell.col] == 0:
                indexes_for_baza.remove(cell)

    # Подсчитываем итоговые затраты
    result = sum(x[cell.row][cell.col] * costs[cell.row][cell.col] for cell in indexes_for_baza)
    print(f"Z = {result} (метод потенциалов)")


def build_path(start_cell, baza_cells):
    # Инициализируем стек с начальной ячейкой и возможными следующими ячейками
    stack = [State(start_cell, 'v',
                   [cell for cell in baza_cells if cell.row == start_cell.row and cell.col != start_cell.col])]

    while stack:
        head = stack[-1]  # Получаем верхний элемент стека

        # Проверяем, не образуется ли замкнутый путь
        if len(stack) >= 4 and ((head.cell.row == start_cell.row) or (head.cell.col == start_cell.col)):
            break

        if not head.next_cells:  # Если нет следующих ячеек, выходим из цикла
            stack.pop()
            continue

        next_cell = head.next_cells.pop()  # Берем следующую ячейку
        next_dir = 'h' if head.prev_dir == 'v' else 'v'  # Меняем направление
        next_cells = [
            cell for cell in baza_cells
            if (cell.col == next_cell.col if next_dir == 'h' else cell.row == next_cell.row)
               and (cell.row != next_cell.row if next_dir == 'h' else cell.col != next_cell.col)
        ]

        stack.append(State(next_cell, next_dir, next_cells))  # Добавляем новую ячейку в стек

    return [state.cell for state in stack]  # Возвращаем путь в виде списка ячеек


# if __name__ == "__main__":
#
#     a = [60, 65, 70]
#     b = [40, 60, 70, 25]
#     costs = [
#         [3, 1, 2, 3],
#         [5, 4, 1, 5],
#         [2, 4, 3, 2]
#     ]
#
#     solve(a, b, costs)

if __name__ == "__main__":

    a = [30, 50, 20]
    b = [15, 15, 40, 30]
    costs = [
        [1, 8, 2, 3],
        [4, 7, 5, 1],
        [5, 3, 4, 4]
    ]

    solve(a, b, costs)
