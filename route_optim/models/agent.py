import numpy as np
import sys
import datetime

def print_progress_bar(iteration, limit):
    fill = '█'
    length = 50
    prefix = 'Episodes: '
    suffix = '(limit: ' + str(limit) + ')'

    percent = ("{0:.1f}").format(100 * (iteration / float(limit)))
    filled_length = int(length * iteration // limit)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}', )
    sys.stdout.flush()

class rl_agent():
    def __init__ (self, env, start_node, end_node, learning_rate, discount_factor, reward_lst):
        # Визначення параметрів навчання
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward_lst = reward_lst

        # Ініціалізація середовища
        self.env = env
        self.env.set_start_end(start_node, end_node)  # задаємо початковий і кінцевий вузли

    # Скидання агента
    def reset(self):
        self.q_table = np.zeros((len(self.env.state_space), len(self.env.action_space)))  # state_space * action_space
        self.logs = {}  # self.logs[episode] = [node_path, edge_path]
        self.best_result = 0

    def act(self):
        pass  # похідні класи визначають це самостійно

    def step(self, action, node_path, edge_path):
        # 0. Ініціалізація кроку
        is_terminate = False
        current_state = node_path[-1]
        current_edge = edge_path[-1] if edge_path else None

        outgoing_edges = self.env.decode_node_to_edges(current_state, direction = 'outgoing')

        # 1. Визначення параметрів нагороди
        invalid_action_reward = self.reward_lst[0]
        dead_end_reward = self.reward_lst[1]
        loop_reward = self.reward_lst[2]
        completion_reward = self.reward_lst[3]
        bonus_reward = self.reward_lst[4]  # = ((self.best_result-current_result)/self.best_result)*100 + 50
        continue_reward = self.reward_lst[5]

        # Визначення нагороди
        reward = continue_reward

        # 2. Розрахунок нагороди та наступного стану
        # Випадок 1. Дія поза межами
        if action not in self.env.decode_edges_to_actions(outgoing_edges):  # наприклад, може бути неможливо повернути праворуч
            reward += invalid_action_reward
            next_state = current_state
            next_edge = current_edge

        # Випадок 2. Допустима дія
        else:
            next_edge = self.env.decode_edges_action_to_edge(outgoing_edges, action)
            next_state = self.env.decode_edge_to_node(next_edge, direction = 'end')

            # Випадок 2-1. Кінцевий вузол
            if next_state in self.env.end_node:
                reward += completion_reward
                is_terminate = True

                # перевірка, чи є маршрут найкоротшим за відстанню/часом
                if self.env.evaluation in ("time"):
                    current_result = self.env.get_edge_time(edge_path + [next_edge]) + self.env.get_tl_offset(edge_path + [next_edge])
                else:
                    current_result = self.env.get_edge_distance(edge_path + [next_edge])

                if self.best_result == 0:
                    self.best_result = current_result
                elif current_result < self.best_result:
                    for edge in edge_path:
                        state_index = self.env.state_space.index(self.env.decode_edge_to_node(edge, direction = 'start'))
                        action_index = self.env.edge_label[edge]
                        self.q_table[state_index][action_index] += bonus_reward
                    self.best_result = current_result

            # Випадок 2-2. Тупиковий маршрут
            elif not self.env.decode_node_to_edges(next_state, direction = 'outgoing'):
                reward += dead_end_reward
                is_terminate = True

                # Повернення назад і знаходження вузького місця
                for edge in reversed(edge_path):
                    if len(self.env.decode_node_to_edges(self.env.decode_edge_to_node(edge, direction = 'end'), direction = 'outgoing')) > 1:
                        break

                    state_index = self.env.state_space.index(self.env.decode_edge_to_node(edge, direction = 'start'))
                    action_index = self.env.edge_label[edge]
                    self.q_table[state_index][action_index] += dead_end_reward

            # Випадок 2-3. Подорож
            elif current_edge != None:
                # Випадок 2-4. Подорож у петлі
                if (current_edge, next_edge) in [(edge_path[i], edge_path[i+1]) for i in range(len(edge_path)-1)]:
                    reward += loop_reward

        return next_edge, next_state, reward, is_terminate  # повертаємо наступний стан, нагороду та is_terminate

    # Оновлення Q-таблиці
    def learn(self, current_state, action, next_state, reward):
        # 1. Отримання оригінального значення Q
        q_predict = self.q_table[self.env.state_space.index(current_state)][action]

        # 2. Розрахунок зміни значення Q
        q_target = reward + self.discount_factor * np.max(self.q_table[self.env.state_space.index(next_state)])
        # що нам потрібно, так це знайти максимальне значення з усіх q_table[next_state][action]

        # 3. Оновлення значення Q практично
        self.q_table[self.env.state_space.index(current_state)][action] += self.learning_rate * (q_target - q_predict)

    # Основна функція, що реалізується
    def train(self, num_episodes, threshold):
        print('Навчання розпочато...')
        start_time = datetime.datetime.now()  # запис часу початку
        self.reset()  # ініціалізація агента

        # Ітерація через епізоди
        for episode in range(num_episodes):

            print_progress_bar(episode, num_episodes)

            # Ініціалізація стану
            node_path = [self.env.start_node]
            edge_path = []
            is_terminate = False

            # Ітерація до досягнення заданого кінця
            while True:
                last_state = node_path[-1]
                if is_terminate or last_state in self.env.end_node:
                    break

                # Вибір дії
                action = self.act(last_state)

                # Виконання дії та спостереження за результатом
                next_edge, next_state, reward, is_terminate = self.step(action, node_path, edge_path)

                # Навчання на основі результату, оновлення Q-таблиці
                self.learn(last_state, action, next_state, reward)

                # Оновлення стану
                if last_state != next_state:  # last_state == next_state тільки якщо дія недійсна
                    edge_path.append(next_edge)
                    node_path.append(next_state)

            # Додавання до журналу
            self.logs[episode] = [node_path, edge_path]

            # Обробка збіжності: > поріг для отримання однакових результатів потрібну кількість разів та впевніться, що досягли кінцевого вузла
            if episode > threshold and self.logs[episode][0][-1] == self.env.end_node:

                # Збіжність, коли час, витрачений у 5 епізодах, є сталим
                threshold_lst = []
                for i in range(threshold):
                    threshold_lst.append(round(self.env.get_edge_time(self.logs[episode-i][1]) + self.env.get_tl_offset(self.logs[episode-i][1]), 2))

                if all(x == threshold_lst[0] for x in threshold_lst):
                    end_time = datetime.datetime.now()  # запис часу закінчення
                    time_difference = end_time - start_time
                    processing_seconds = time_difference.total_seconds()

                    # --- результати ---
                    print('\nНавчання завершено...\n')
                    print(f'-- Останній епізод: {episode}\n')
                    print(f'-- Вузли: {self.logs[episode][0]}\n')
                    print(f'-- Ребра: {self.logs[episode][1]}\n')
                    print(f'-- Час обробки: {processing_seconds} секунд')

                    if self.env.evaluation in ("time"):
                        print(f'-- Час подорожі: {round((self.env.get_edge_time(self.logs[episode][1]) + self.env.get_tl_offset(self.logs[episode][1]))/60, 2)} хвилин')
                    else:
                        print(f'-- Дистанція подорожі: {round(self.env.get_edge_distance(self.logs[episode][1]), 2)} м')

                    return self.logs[episode][0], self.logs[episode][1], episode, self.logs

            # Обробка випадку, коли не вдалося досягти збіжності
            if episode + 1 == num_episodes:
                print('\nНавчання завершено...\n')
                end_time = datetime.datetime.now()
                time_difference = end_time - start_time
                processing_seconds = time_difference.total_seconds()
                print(f'-- Час обробки: {processing_seconds} секунд')
                self.env.plot_performance(episode, self.logs)  # все одно відображаємо plot_performance, навіть якщо не вдалося досягти збіжності
                sys.exit(f'Не вдалося знайти найкоротший маршрут протягом {num_episodes} епізодів')

class Q_Learning(rl_agent):
    def __init__ (self, env, start_node, end_node):
        # --------------------------
        # Гіперпараметри
        # --------------------------
        learning_rate = 0.9  # alpha
        discount_factor = 0.1  # gamma
        reward_lst = [-50, -50, -30, 50, 50, 0]
        # --------------------------
        #
        # --------------------------

        """
        Гіперпараметри алгоритму:
        - learning_rate (float):
        - discount_factor (float):
            Q(S,a) = Q(S,a) + alpha * (R + gamma * max(Q(S',a') - Q(S,a))
        - reward_lst (list [6])
            0. invalid_action_reward: дія не дозволена, за замовчуванням -50
            1. dead_end_reward: зустріч з тупиком, за замовчуванням -50
            2. loop_reward: створення петлі, за замовчуванням -30
            3. completion_reward, за замовчуванням 50
            4. bonus_reward: найкоротший маршрут на цей момент, за замовчуванням 50
            5. continue_reward: стимулювання агента йти прямо, за замовчуванням 0
        """

        super().__init__(env, start_node, end_node, learning_rate, discount_factor, reward_lst)

    def act(self, state):
        # Вибір дії з найвищим значенням Q
        state_index = self.env.state_space.index(state)
        action = np.argmax(self.q_table[state_index])
        return action

class SARSA(rl_agent):
    def __init__ (self, env, start_node, end_node):
        # --------------------------
        # Гіперпараметри
        # --------------------------
        learning_rate = 0.9  # alpha
        discount_factor = 0.1  # gamma
        exploration_rate = 0.1  # співвідношення дослідження та використання
        reward_lst = [-50, -50, -30, 50, 50, 0]  # подібно до Q_Learning
        # --------------------------
        #
        # --------------------------

        super().__init__(env, start_node, end_node, learning_rate, discount_factor, reward_lst)
        self.exploration_rate = exploration_rate

    def act(self, state):
        if np.random.random() < self.exploration_rate:
            # Дослідження
            action = np.random.choice(len(self.env.action_space))
        else:
            # Використання
            state_index = self.env.state_space.index(state)
            action = np.argmax(self.q_table[state_index])
        return action
