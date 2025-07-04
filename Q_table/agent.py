import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma  # 衰减系数 (discount factor)
        self.epsilon = 0.5  # 初始探索率
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        self.sample_count += 1
        # Epsilon-greedy 策略：以 epsilon 的概率进行探索，以 1-epsilon 的概率进行利用
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_dim)  # 随机探索选取一个动作
        else:
            action = np.argmax(self.Q_table[state])  # 根据Q表格选择最优动作

        self.epsilon = max(0.01, self.epsilon * 0.995)

        return action

    def update(self, state, action, reward, next_state, done):

        # Q-learning 更新规则：Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s',a')) - Q(s,a)]

        if done:
            td_target = reward  # 回合结束，目标值只有当前奖励
        else:
            # TD 目标：当前奖励 + 折扣因子 * 下一个状态的最大 Q 值
            td_target = reward + self.gamma * np.max(self.Q_table[next_state])

        # TD 误差：TD 目标 - 当前 Q(s,a)c
        td_error = td_target - self.Q_table[state, action]

        # 更新 Q 值
        self.Q_table[state, action] += self.lr * td_error

    def predict(self, state):
        ############################ 可根据自己需求更改##################################
        if type(state) != int:
            state = state[0]
        action = np.argmax(self.Q_table[state])
        return action

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")