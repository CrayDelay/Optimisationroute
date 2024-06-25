import heapq
import datetime

class Dijkstra:
    def __init__ (self, env, start_node, end_node):
        self.env = env
        self.env.set_start_end(start_node, end_node)  # виклик set_start_end() у env для встановлення початкового та кінцевого вузлів

    def reset(self):
        self.cost = {node: float('inf') for node in self.env.nodes}  # встановлюємо вартість до нескінченності
        self.predecessor = {node: None for node in self.env.nodes}
        self.cost[self.env.start_node] = 0  # вартість від початкового вузла до самого себе дорівнює 0
        self.priority_queue = [(0, self.env.start_node)]  # ініціалізація черги пріоритетів

    def calculate_cost(self, current_cost, adj_edge):
        if self.env.evaluation in ("time"):
            cost = current_cost + self.env.get_edge_time(adj_edge)  # обчислення вартості за часом
        else:  # in ("distance")
            cost = current_cost + self.env.get_edge_distance(adj_edge)  # обчислення вартості за відстанню
        return cost

    # основна функція в алгоритмі Дейкстри
    def search(self):
        print('Пошук починається...')
        start_time = datetime.datetime.now()  # фіксація часу початку

        self.reset()  # початковий стан алгоритму

        while self.priority_queue:
            current_cost, current_node = heapq.heappop(self.priority_queue)  # отримуємо вузол з мінімальною вартістю з купи

            # Якщо вузол є кінцевим вузлом, то припиняємо пошук
            if current_node == self.env.end_node:
                break

            # Досліджуємо сусідні вузли
            for adj_edge in self.env.decode_node_to_edges(current_node, direction='outgoing'):
                # Відповідний сусідній вузол
                adj_node = self.env.decode_edge_to_node(adj_edge, direction='end')
                # Обчислення вартості сусіда
                temp_cost = self.calculate_cost(current_cost, adj_edge)

                # Якщо попередня відстань менша за поточну відстань сусіда
                if temp_cost < self.cost[adj_node]:
                    self.cost[adj_node] = temp_cost  # оновлення відстані сусіда
                    self.predecessor[adj_node] = current_node  # оновлення попередника сусіда
                    heapq.heappush(self.priority_queue, (temp_cost, adj_node))  # додавання сусіда до черги пріоритетів

        # Побудова шляху від початкового вузла до цільового вузла
        node_path = []
        edge_path = []

        temp_node = self.env.end_node
        while temp_node is not None:
            node_path.append(temp_node)
            temp_node = self.predecessor[temp_node]
        node_path.reverse()

        for index in range(len(node_path)-1):
            intersect_edge = set(self.env.decode_node_to_edges(node_path[index], "outgoing")) & set(self.env.decode_node_to_edges(node_path[index+1], "incoming"))  # знаходимо ребро між двома вузлами
            edge_path.append(next(iter(intersect_edge)))  # intersect_edge є множиною, тому перетворюємо її на ітератор і беремо наступний елемент

        # фіксація часу закінчення пошуку
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        processing_seconds = time_difference.total_seconds()  # отримуємо час у секундах

        # --- результати ---
        print('Пошук завершено...\n')
        print(f'-- Вузли: {node_path}\n')  # список відвіданих вузлів
        print(f'-- Ребра: {edge_path}\n')  # список відвіданих ребер
        print(f'-- Час обробки: {processing_seconds} секунд')

        if self.env.evaluation in ("time"):
            print(
                f'-- Час подорожі: {round((self.env.get_edge_time(edge_path) + self.env.get_tl_offset(edge_path)) / 60, 2)} хвилин')
        else:  # in ("distance")
            print(f'-- Дистанція подорожі: {round(self.env.get_edge_distance(edge_path), 2)} м')

        return node_path, edge_path  # повертаємо знайдений шлях у вигляді списків вузлів і ребер

