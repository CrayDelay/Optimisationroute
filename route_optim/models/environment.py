import sys
import sumolib
import math
import networkx as nx
import matplotlib.pyplot as plt
import random

class traffic_env:
    def __init__ (self, network_file, tls, congestion=[], evaluation="", congestion_level=""):
        # 1. Визначення network_file
        self.network_file = network_file  # зчитування файлу

        self.net = sumolib.net.readNet(network_file)  # файл -> net
        self.nodes = [node.getID().upper() for node in self.net.getNodes()]  # net -> вузли (ID)
        self.edges = [edge.getID() for edge in self.net.getEdges()]  # net -> ребра (ID)

        self.tls = tls  # [tl_id][link_index]=[90] (dict)
        self.tls_space = [tl.getID() for tl in self.net.getTrafficLights()]
        self.tls_meet = []  # для відображення на карті
        self.congestion_meet = []  # для відображення на карті

        self.action_space = [0, 1, 2, 3]  # action_space
        self.state_space = self.nodes  # state_space
        self.edge_label = self.decode_edges_to_label()  # надання кожному ребру мітки на основі їх напрямку з точки зору координат x-y


        # 2. Визначення заторів на ребрах з оригінальним паттерном
        if congestion:  # якщо затори визначено
            self.congested_edges = [item[0] for item in congestion]
            self.congestion_duration = [item[1] for item in congestion]  # тривалість затору

            for edge in self.congested_edges:  # перевірка, чи всі затори на ребрах знаходяться в мережі
                if edge not in self.edges:
                    sys.exit(f'Error: Invalid congestion_edges {edge}')
            # print(f'Congested Edges: {list(zip(self.congested_edges, self.congestion_duration))}')
            # print(f'Congested/Total: {len(self.congested_edges)}/{len(self.edges)}')

        else:  # якщо затори не визначено, то налаштування ребер та їх тривалості випадковим чином
            if congestion_level == "low":
                traffic_level = 0.05  # 5% заторів
            elif congestion_level == "medium":
                traffic_level = 0.10  # 10% заторів
            elif congestion_level == "high":
                traffic_level = 0.20  # 20% заторів
            self.congested_edges = random.sample(self.edges, round(len(self.edges) * traffic_level))
            self.congestion_duration = [random.randint(60, 120) for _ in range(len(self.congested_edges))]  # 1~2 хв
            # print(f'Congested Edges: {list(zip(self.congested_edges, self.congestion_duration))}')
            # print(f'Congested/Total: {len(self.congested_edges)}/{len(self.edges)}')


        # 3. Визначення типу оцінювання
        if evaluation not in ('distance', 'time'):
            sys.exit('Error: Invalid evaluation type, provide only "distance" or "time"')
        self.evaluation = evaluation


    # Встановлення початкових та кінцевих вузлів
    def set_start_end(self, start_node, end_node):


        # Перевірка, чи вузли є дійсними
        if start_node not in self.nodes:
            sys.exit('Error: Invalid start node')
        elif end_node not in self.nodes:
            sys.exit('Error: Invalid end node')
        else:
            self.start_node = start_node
            self.end_node = end_node


    # Зіставлення вузлів з ребрами
    def decode_node_to_edges(self, node, direction=None):


        # Перевірка, чи напрямок є дійсним
        if direction not in ('incoming', 'outgoing', None):
            sys.exit(f'Invalid direction: {direction}')

        edges = []
        net_node = self.net.getNode(node)

        # Зіставлення вузлів та напрямку для повернення ребер
        if direction == 'incoming':
            for edge in net_node.getIncoming():
                if edge.getToNode().getID() == node:
                    edges.append(edge.getID())

        elif direction == 'outgoing':
            for edge in net_node.getOutgoing():
                if edge.getFromNode().getID() == node:
                    edges.append(edge.getID())

        else:
            for edge in net_node.getIncoming() + net_node.getOutgoing():
                if edge.getToNode().getID() == node or edge.getFromNode().getID() == node:
                    edges.append(edge.getID())

        return edges


    # Маркування ребер на основі перехресть (0 Право -> 1 Вгору -> 2 Ліво -> 3 Вниз)
    def decode_edges_to_label(self):

        edge_labelled = {edge: None for edge in self.edges}

        def get_edge_label(node, outgoing_edges):
            # отримання вихідних ребер вузла
            start_x, start_y = self.net.getNode(node).getCoord()

            # збереження кутів ребер
            edge_angle = []

            # для кожного вихідного ребра
            for edge in outgoing_edges:
                # отримання кінцевого вузла ребра
                end_node = self.decode_edge_to_node(edge)
                end_x, end_y = self.net.getNode(end_node).getCoord()

                # отримання delta_x та delta_y
                delta_x = end_x - start_x
                delta_y = end_y - start_y

                # отримання кута за delta_x та delta_y
                angle = math.degrees(math.atan2(delta_y, delta_x))

                # збереження ребра та його відповідного кута
                edge_angle.append((edge, angle))

            # сортування від 0 до 180 до -180 до 0 (Право -> Вгору -> Ліво -> Вниз -> Право)
            edge_angle = sorted(edge_angle, key=lambda x: ((x[1] >= 0) * -180, x[1]))

            # маркування ребер
            for i in range(len(edge_angle)):
                edge_labelled[edge_angle[i][0]] = i  # edge_angle[i][0] - це впорядковане ребро, а [1] - його кут

        for node in self.nodes:
            outgoing_edges = self.decode_node_to_edges(node, 'outgoing')
            if outgoing_edges:
                get_edge_label(node, outgoing_edges)

        return edge_labelled  # зверніть увагу, що два ребра в протилежних напрямках мають різні ID (розглядаються як різні ребра)


    # Знаходження дій для заданих ребер
    def decode_edges_to_actions(self, edges):


        # Перевірка, чи ребра є в списку ребер
        for edge in edges:
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space')

        # Отримання мітки кожного ребра
        edge_label = self.edge_label

        # Повертає список дій
        actions_lst = []
        for action in self.action_space:
            if action in [edge_label[edge] for edge in edges]:
                actions_lst.append(action)
        return actions_lst


    # Знаходження відповідного ребра за заданим набором ребер від вузла та дією
    def decode_edges_action_to_edge(self, edges, action):


        # Перевірка, чи ребра є в списку ребер
        for edge in edges:
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space')

        # Отримання напрямку кожного ребра
        edge_label = self.edge_label

        for edge in edges:
            if edge_label[edge] == action:
                return edge
        return None


    # Знаходження кінцевого вузла за заданим ребром
    def decode_edge_to_node(self, search_edge, direction='end'):

        # Перевірка, чи ребра є в списку ребер
        if search_edge not in self.edges:
            sys.exit('Error: Edge not in Edges Space!')

        edge = self.net.getEdge(search_edge)

        if direction == 'start':
            node = edge.getFromNode().getID()

        elif direction == 'end':
            node = edge.getToNode().getID()

        return node


    # Знаходження загальної відстані, пройденої по заданому ребру / реберному шляху
    def get_edge_distance(self, travel_edges):


        total_distance = 0

        if isinstance(travel_edges, str):  # перетворення "" у [""]
            travel_edges = [travel_edges]

        for edge in travel_edges:
            # Перевірка, чи ребра є в списку ребер
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space ...call by get_edge_distance')
            # Підсумування відстані кожного ребра
            total_distance += self.net.getEdge(edge).getLength()

        return total_distance


    # Знаходження загального часу, витраченого на задане ребро / реберний шлях
    def get_edge_time(self, travel_edges):

        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        total_time = 0
        for edge in travel_edges:
            # Перевірка, чи ребра є в списку ребер
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space ...call by get_edge_time')
            # Підсумування відстані кожного ребра
            total_time += self.net.getEdge(edge).getLength() / self.net.getEdge(edge).getSpeed()

        # покарання за час маршруту
        for i in range(len(travel_edges)):  # покарання за час на конкретному ребрі через затори
            if travel_edges[i] in self.congested_edges:
                total_time += self.congestion_duration[self.congested_edges.index(travel_edges[i])]

        return total_time


    # Знаходження затримки часу, викликаної світлофором
    def get_tl_offset(self, travel_edges):

        self.tls_meet = []  # для відображення на карті
        self.congestion_meet = []  # для відображення на карті

        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        current_time = 0
        for edge in range(len(travel_edges) - 1):
            current_edge = travel_edges[edge]
            next_edge = travel_edges[edge+1]

            # 0. Перевірка, чи ребра є в списку ребер
            if current_edge not in self.edges:
                sys.exit(f'Error: Edge {current_edge} not in Edges Space ...call by get_tl_offset')

            if current_edge in self.congested_edges and current_edge not in self.congestion_meet:
                self.congestion_meet.append(current_edge)  # для відображення на карті

            # 1. Підсумування відстані кожного ребра
            current_time += self.net.getEdge(current_edge).getLength() / self.net.getEdge(current_edge).getSpeed()

            # 2. Знаходження кінцевої точки ребра
            tl = self.net.getEdge(current_edge).getToNode().getID()
            if tl not in self.tls_space:
                continue

            self.tls_meet.append(tl)  # для відображення на карті

            # 3. Знаходження з'єднання між current_edge і next_edge
            connections_set = self.net.getTLS(tl).getConnections()

            for j in range(len(connections_set)):
                connection = connections_set[j]
                if connection[0] in self.net.getEdge(current_edge).getLanes() and connection[1] in self.net.getEdge(next_edge).getLanes():
                    break

            # 4. Визначення, що це з'єднання є nth link цього tl_node
            tl_phase = self.tls[tl][connection[2]]

            idle_time = 0
            if tl_phase[int(current_time) % 90] != "r":
                continue
            else:
                for phase_index in range(90):
                    if tl_phase[(int(current_time) + phase_index) % 90] != "r":
                        idle_time = phase_index
                        break

            # 5. Підсумування idle
            current_time += idle_time

        return current_time - self.get_edge_time(travel_edges)


    # ------ Візуалізація графа ------
    def plot_visualised_result(self, travel_edges):


        nodes_dict = {}  # список x_coord та y_coord кожного вузла
        for node in self.nodes:
            x_coord, y_coord = self.net.getNode(node).getCoord()
            nodes_dict[node] = (x_coord, y_coord)

        edges_dict = {}  # список від_point та to_point кожного ребра
        for edge in self.edges:
            from_id = self.net.getEdge(edge).getFromNode().getID()
            to_id = self.net.getEdge(edge).getToNode().getID()
            edges_dict[edge] = (from_id, to_id)

        # Малювання розташування мережі
        net_G = nx.Graph()
        for edge in edges_dict:
            net_G.add_edge(edges_dict[edge][0], edges_dict[edge][1])
        pos = {node: nodes_dict[node] for node in nodes_dict}
        nx.draw(
            net_G, pos, with_labels=False,
            node_color='DimGray', node_size=10,
            edge_color='DarkGray'
        )

        # Малювання вибраного маршруту
        route_G = nx.Graph()
        for edge in travel_edges:
            route_G.add_edge(edges_dict[edge][0], edges_dict[edge][1])
        nx.draw(
            route_G, pos, with_labels=False,
            node_color='SeaGreen', node_size=30,
            edge_color='MediumSeaGreen', width=3,
            arrows=True, arrowsize=7, arrowstyle='-|>'
        )

        # Малювання вузлів світлофорів та заторів у оцінці часу
        if self.evaluation in ("time"):
            nx.draw_networkx_nodes(
                net_G, pos,
                nodelist=self.tls_meet, node_color='Crimson', node_size=30
            )

            congested_lst = [edges_dict[edge] for edge in self.congested_edges]  # затори на ребрах
            nx.draw_networkx_edges(
                net_G, pos,
                edgelist=congested_lst, edge_color='Gold', width=3
            )

            congestion_meet = [edges_dict[edge] for edge in self.congestion_meet]  # затори на ребрах
            nx.draw_networkx_edges(
                net_G, pos,
                edgelist=congestion_meet, edge_color='IndianRed', width=3
            )

        plt.show()


    def plot_performance(self, num_episodes, logs):

        plt.title("Performance of Agent")
        plt.xlabel("Episode")
        if self.evaluation in ("time"):
            plt.ylabel("Time")
            evaluation = [(self.get_edge_time(logs[episode][1]) + self.get_tl_offset(logs[episode][1]))/60  for episode in range(num_episodes)]
        else:
            plt.ylabel("Distance")
            evaluation = [self.get_edge_distance(logs[episode][1]) for episode in range(num_episodes)]
        plt.plot(range(num_episodes), evaluation)
        plt.show()
