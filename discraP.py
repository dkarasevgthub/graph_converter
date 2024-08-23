import os
import platform


# Утилиты для управления консолью
def clear_console():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def set_console_size(width, height):
    if platform.system() == "Windows":
        os.system(f"mode con: cols={width} lines={height}")
    else:
        os.system(f"printf'\e[8;{height};{width}t'")


# Приветственное сообщение
def welcome_message():
    clear_console()
    set_console_size(140, 40)
    print("""
     ________        ___      .______     ____    ____      ___      
    |       /       /   \     |   _  \    \   \  /   /     /   \     
    `---/  /       /  ^  \    |  |_)  |    \   \/   /     /  ^  \    
       /  /       /  /_\  \   |      /      \_    _/     /  /_\  \   
      /  /----.  /  _____  \  |  |\  \--.     |  |      /  _____  \  
     /________| /__/     \__\ | _| `.___|     |__|     /__/     \__\ 

    """)


# Функции для ввода различных видов графов
def input_adjacency_matrix():
    def is_weighted(matrix):
        for row in matrix:
            for value in row:
                if value != 0 and value != 1:
                    return True
        return False

    matrix = []
    print("Введите строки матрицы смежности, разделяя элементы пробелами. Для завершения ввода введите пустую строку.")
    while True:
        row = input()
        if row == "":
            break
        matrix.append([int(x) for x in row.split()])
    if not matrix:
        raise ValueError("Матрица смежности не может быть пустой.")
    if is_weighted(matrix):
        print("Граф взвешенный.")
    else:
        print("Граф невзвешенный.")
    return matrix


def input_incidence_matrix():
    def is_weighted(matrix):
        for row in matrix:
            for value in row:
                if value != 0 and value != 1:
                    return True
        return False

    matrix = []
    print(
        "Введите строки матрицы инцидентности, разделяя элементы пробелами. Для завершения ввода введите пустую строку.")
    while True:
        row = input()
        if row == "":
            break
        matrix.append([int(x) for x in row.split()])
    if not matrix:
        raise ValueError("Матрица инцидентности не может быть пустой.")
    if is_weighted(matrix):
        print("Граф взвешенный.")
    else:
        print("Граф невзвешенный.")
    return matrix


def input_adjacency_list():
    def is_weighted(adj_list):
        for vertex, neighbors in adj_list.items():
            for neighbor in neighbors:
                if isinstance(neighbor, tuple) and len(neighbor) > 1:
                    return True
        return False

    adj_list = {}
    print(
        "Введите список смежности. Для каждой вершины введите список соседних вершин и их вес (если есть), разделяя элементы пробелами. Для завершения ввода введите пустую строку.")
    while True:
        line = input()
        if line == "":
            break
        vertex, *neighbors = line.split()
        adj_list[int(vertex)] = [(int(nbr[:-1]), int(nbr[-1])) if len(nbr) > 1 else int(nbr) for nbr in neighbors]
    if not adj_list:
        raise ValueError("Список смежности не может быть пустым.")
    if is_weighted(adj_list):
        print("Граф взвешенный.")
    else:
        print("Граф невзвешенный.")
    return adj_list


def input_unsorted_edge_list():
    def is_weighted(edge_list):
        for edge in edge_list:
            if len(edge) >= 2:  # Проверяем, есть ли вес у ребра
                return True
        return False

    edge_list = []
    print(
        "Введите список ребер (дуг) графа, разделяя вершины и их вес (если есть) пробелами. Для завершения ввода введите пустую строку.")
    while True:
        line = input()
        if line == "":
            break
        edge = tuple(map(int, line.split()))
        edge_list.append(edge)
    if not edge_list:
        raise ValueError("Список ребер (дуг) не может быть пустым.")
    if is_weighted(edge_list):
        print("Граф взвешенный.")
    else:
        print("Граф невзвешенный.")
    return edge_list


def input_sorted_edge_list():
    def is_weighted(edge_list):
        for edge in edge_list:
            if len(edge) >= 2:
                return True
        return False

    edge_list = []
    print(
        "Введите отсортированный список ребер (дуг) графа. Для каждой дуги введите начальную вершину (i), конечную вершину (j) и их вес (если есть) через пробел. Для завершения ввода введите пустую строку.")
    print("I J C")
    while True:
        line = input()
        if line == "":
            break
        edge = tuple(map(int, line.split()))
        edge_list.append(edge)
    if not edge_list:
        raise ValueError("Список ребер (дуг) не может быть пустым.")
    if is_weighted(edge_list):
        print("Граф взвешенный.")
    else:
        print("Граф невзвешенный.")
    return edge_list


def input_bundle_edge_list():
    def is_weighted(edge_list):
        for bundle in edge_list:
            for edge in bundle:
                if len(edge) > 2:
                    return True
        return False

    edge_list = []
    print(
        "Введите список пучков ребер (дуг) графа. Каждый пучок должен быть введен отдельной строкой, разделяя начальную вершину (i), конечную вершину (j), дополнительную информацию (k) и голову списка (h), если она есть, пробелами. Для завершения ввода введите пустую строку.")
    print("I J K H")
    while True:
        bundle = []
        while True:
            line = input()
            if line == "":
                break
            edge = tuple(map(int, line.split()))
            bundle.append(edge)
        if not bundle:
            break
        edge_list.append(bundle)
    if not edge_list:
        raise ValueError("Список пучков ребер (дуг) не может быть пустым.")
    if is_weighted(edge_list):
        print("Граф взвешенный.")
    else:
        print("Граф невзвешенный.")
    return edge_list


# Функции для преобразования графов
def incidence_matrix_to_adjacency_matrix(incidence_matrix):
    num_vertices = len(incidence_matrix)
    num_edges = len(incidence_matrix[0])
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    for edge_idx in range(num_edges):
        connected_vertices = []
        for vertex_idx in range(num_vertices):
            if incidence_matrix[vertex_idx][edge_idx] == 1:
                connected_vertices.append(vertex_idx)
        if len(connected_vertices) == 2:
            vertex1, vertex2 = connected_vertices
            adjacency_matrix[vertex1][vertex2] = 1
            adjacency_matrix[vertex2][vertex1] = 1
    return adjacency_matrix


def adjacency_list_to_adjacency_matrix(adjacency_list):
    num_vertices = len(adjacency_list)
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    for vertex, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            if isinstance(neighbor, tuple):
                neighbor_vertex, weight = neighbor
                adjacency_matrix[vertex][neighbor_vertex] = weight
            else:
                adjacency_matrix[vertex][neighbor] = 1
    return adjacency_matrix


def edge_list_to_adjacency_matrix(edge_list):
    num_vertices = max(max(edge) for edge in edge_list) + 1
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    for edge in edge_list:
        if len(edge) >= 2:  # Проверяем, есть ли вес у ребра
            vertex1, vertex2, weight = edge
            adjacency_matrix[vertex1][vertex2] = weight
            adjacency_matrix[vertex2][vertex1] = weight
        else:
            vertex1, vertex2 = edge
            adjacency_matrix[vertex1][vertex2] = 1
            adjacency_matrix[vertex2][vertex1] = 1
    return adjacency_matrix


def bundle_edge_list_to_adjacency_matrix(bundle_edge_list):
    num_vertices = len(bundle_edge_list)
    H = [-1] * num_vertices  # Массив голов списков
    L = [-1] * sum(len(bundle) for bundle in bundle_edge_list)  # Массив ссылок на следующие дуги
    I = []
    J = []
    K = []
    C = []
    current_edge_idx = 0
    for vertex, bundle in enumerate(bundle_edge_list):
        for edge in bundle:
            H[vertex] = current_edge_idx
            if len(edge) > 2:  # Если дуга имеет дополнительную информацию
                tail, head, weight, next_edge = edge
                K.append(tail)
                C.append(weight)
            else:
                tail, head, next_edge = edge
                K.append(tail)
                C.append(1)  # Если вес не указан, считаем его равным 1
            I.append(tail)
            J.append(head)
            L[current_edge_idx] = next_edge
            current_edge_idx += 1
    # Инициализируем матрицу смежности
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    # Заполняем матрицу смежности на основе массивов H и L
    for vertex, head_edge in enumerate(H):
        current_edge = head_edge
        while current_edge != -1:
            adjacency_matrix[I[current_edge]][J[current_edge]] = C[current_edge]
            current_edge = L[current_edge]
    print("I:", I)
    print("J:", J)
    print("K:", K)
    print("C:", C)
    return adjacency_matrix


def adjacency_matrix_to_incidence_matrix(adjacency_matrix):
    num_vertices = len(adjacency_matrix)
    num_edges = sum(sum(row) for row in adjacency_matrix) // 2  # Учитываем, что граф ненаправленный
    incidence_matrix = [[0] * num_edges for _ in range(num_vertices)]
    edge_index = 0
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):  # Перебираем только верхний треугольник матрицы
            weight = adjacency_matrix[i][j]
            if weight != 0:
                incidence_matrix[i][edge_index] = weight
                incidence_matrix[j][edge_index] = -weight
                edge_index += 1
    return incidence_matrix


def adjacency_matrix_to_adjacency_list(adjacency_matrix):
    adjacency_list = []
    for i in range(len(adjacency_matrix)):
        neighbors = []
        for j in range(len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] != 0:
                neighbors.append(j)
        adjacency_list.append(neighbors)
    return adjacency_list


def adjacency_matrix_to_unsorted_edge_list(adjacency_matrix, sorted=False):
    edge_list = []
    c = []  # Массив для хранения дополнительной информации
    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] != 0:
                if sorted:
                    edge_list.append((i, j, adjacency_matrix[i][j]))
                else:
                    edge_list.append((i, j))
                # Заполняем массив c дополнительной информацией (если нужно)
                # В данном примере просто добавляем вес ребра
                c.append(adjacency_matrix[i][j])
    return edge_list, c


def adjacency_matrix_to_sorted_edge_list(adjacency_matrix, sorted=True):
    edge_list = []
    c = []  # Массив для хранения дополнительной информации
    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] != 0:
                if sorted:
                    edge_list.append((min(i, j), max(i, j), adjacency_matrix[i][j]))
                else:
                    edge_list.append((i, j))
                # Заполняем массив c дополнительной информацией (если нужно)
                # В данном примере просто добавляем вес ребра
                c.append(adjacency_matrix[i][j])
    if sorted:
        edge_list.sort()
    return edge_list, c


def adjacency_matrix_to_bundle_edge_list(adjacency_matrix):
    num_vertices = len(adjacency_matrix)
    H = [-1] * num_vertices  # Массив голов списков
    L = []  # Массив ссылок на следующие дуги
    I = []
    J = []
    K = []
    for i in range(num_vertices):
        for j in range(num_vertices):
            if adjacency_matrix[i][j] != 0:
                if H[i] == -1:  # Если это первая дуга из вершины i
                    H[i] = len(L)
                else:
                    L.append(len(L))
                L.append(-1)  # Добавляем фиктивный номер в конец списка
                I.append(i)
                J.append(j)
                K.append(adjacency_matrix[i][j])

    return I, J, K, H


# Ввод исходного графа пользователем
def input_graph():
    print("Выберите формат ввода графа:")
    print("1: Матрица смежности")
    print("2: Матрица инцидентности")
    print("3: Список смежности")
    print("4: Неотсортированный список дуг")
    print("5: Отсортированный список дуг")
    print("6: Список пучков дуг")
    choice = int(input("Введите цифру выбора: "))
    if choice == 1:
        return "adjacency_matrix", input_adjacency_matrix()
    elif choice == 2:
        return "incidence_matrix", input_incidence_matrix()
    elif choice == 3:
        return "adjacency_list", input_adjacency_list()
    elif choice == 4:
        return "unsorted_edge_list", input_unsorted_edge_list()
    elif choice == 5:
        return "sorted_edge_list", input_sorted_edge_list()
    elif choice == 6:
        return "bundle_edge_list", input_bundle_edge_list()
    else:
        print("Неправильный выбор")
        return input_graph()


# Преобразование графа в матрицу смежности
def convert_to_adjacency_matrix(graph, input_type):
    if input_type == "adjacency_matrix":
        return graph
    elif input_type == "incidence_matrix":
        return incidence_matrix_to_adjacency_matrix(graph)
    elif input_type == "adjacency_list":
        return adjacency_list_to_adjacency_matrix(graph)
    elif input_type == "unsorted_edge_list":
        return edge_list_to_adjacency_matrix(graph)
    elif input_type == "sorted_edge_list":
        return edge_list_to_adjacency_matrix(graph)
    elif input_type == "bundle_edge_list":
        return bundle_edge_list_to_adjacency_matrix(graph)
    else:
        raise ValueError("Неправильный тип ввода")


# Выбор вида выходного графа и преобразование
def output_graph(adjacency_matrix):
    print("Выберите формат вывода графа:")
    print("1: Матрица смежности")
    print("2: Матрица инцидентности")
    print("3: Список смежности")
    print("4: Неотсортированный список дуг")
    print("5: Отсортированный список дуг")
    print("6: Список пучков дуг")
    choice = int(input("Введите цифру выбора: "))
    if choice == 1:
        print(adjacency_matrix)
    elif choice == 2:
        print(adjacency_matrix_to_incidence_matrix(adjacency_matrix))
    elif choice == 3:
        print(adjacency_matrix_to_adjacency_list(adjacency_matrix))
    elif choice == 4:
        print(adjacency_matrix_to_unsorted_edge_list(adjacency_matrix, sorted=False))
    elif choice == 5:
        print(adjacency_matrix_to_sorted_edge_list(adjacency_matrix, sorted=True))
    elif choice == 6:
        print(adjacency_matrix_to_bundle_edge_list(adjacency_matrix))
    else:
        print("Неправильный выбор")
        output_graph(adjacency_matrix)


if __name__ == "__main__":
    welcome_message()
    input_type, graph = input_graph()
    adjacency_matrix = convert_to_adjacency_matrix(graph, input_type)
    output_graph(adjacency_matrix)
