import os, sys
import xml.etree.ElementTree as ET

from models import environment
from models import agent
from models import dijkstra

def sumo_config():
    os.environ["SUMO_HOME"] = '$SUMO_HOME'

    # Перевірка, чи SUMO успішно налаштований
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("будь ласка, оголосіть змінну середовища 'SUMO_HOME'")

def tls_from_tllxml(file_name):

    tree = ET.parse(file_name)
    root = tree.getroot()

    tls_data = {}
    for tl in root.findall('.//tlLogic'):
        tl_id = tl.get('id')

        if tl_id not in tls_data:
            tls_data[tl_id] = {}
        else:
            sys.exit(f"Error: {tl_id} duplicated")

        phases = tl.findall('.//phase')

        for phase in phases:
            duration = phase.get('duration')
            state = phase.get('state')

            # Створення tls_data[tl_id][link_index] = n-ий символ у стані, що повторюється duration разів
            for link_index in range(len(state)):
                if link_index not in tls_data[tl_id]:
                    tls_data[tl_id][link_index] = []
                tls_data[tl_id][link_index] += [state[link_index] for _ in range(int(duration))]

    return tls_data

if __name__ == '__main__':

    # 01 Налаштування SUMO
    sumo_config()

    # 02 Налаштування змінних мережі
    network_file = './network_files/ncku_network.net.xml'
    tls = tls_from_tllxml('./network_files/ncku_network.tll.xml')  # tll.xml експортується з netedit
    congestion = []  # можна визначити, але якщо порожній, env випадково вирішить, які краї перевантажені
    start_node = "864831599"  # можна визначити, діапазон - вузли у мережі
    end_node = "5739293224"

    # 03 Ініціація середовища
    env = environment.traffic_env(
        network_file=network_file,
        tls=tls,
        congestion=congestion,
        evaluation="time",  # Тип: "destination" | "time"
        congestion_level="low",  # Тип: "low" | "medium" | "high", застосовується лише якщо congestion не визначено
    )

    # 04 Активація агента
    # -------------------
    # Алгоритм Дейкстри
    # -------------------
    print(f"\nDijkstra Algorithm{'.' * 100}")
    Dijkstra = dijkstra.Dijkstra(env, start_node, end_node)
    node_path, edge_path = Dijkstra.search()
    env.plot_visualised_result(edge_path)

    # -------------------
    # Алгоритм Q_Learning
    # -------------------
    print(f"\nQ_Learning Algorithm{'.' * 100}")
    QLearning_agent = agent.Q_Learning(env, start_node, end_node)
    node_path, edge_path, episode, logs = QLearning_agent.train(5000, 5)  # ліміт епізодів, поріг для конвергенції
    env.plot_performance(episode, logs)
    env.plot_visualised_result(edge_path)

    # -------------------
    # Алгоритм SARSA
    # -------------------
    print(f"\nSARSA Algorithm{'.' * 100}")
    SARSA_agent = agent.SARSA(env, start_node, end_node)
    node_path, edge_path, episode, logs = SARSA_agent.train(5000, 20)  # ліміт епізодів, поріг для конвергенції
    env.plot_performance(episode, logs)
    env.plot_visualised_result(edge_path)
