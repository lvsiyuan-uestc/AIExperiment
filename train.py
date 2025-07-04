import sys
print("Python interpreter path:", sys.executable)
print("Python search paths (sys.path):")
for p in sys.path:
    print(f"  {p}")

import gym
print("Gym module loaded from:", gym.__file__)
import numpy
print("Numpy module loaded from:", numpy.__file__)

# 你可以继续你的训练代码...

import os
import gym
import torch

from utils import CliffWalkingWapper, save_results, make_dir
from agent import QLearning
from plot import plot_rewards

curr_path = os.path.dirname(__file__)


class QlearningConfig:
    '''训练相关参数'''

    def __init__(self):
        self.seed = 0
        self.algo = 'Qlearning'
        self.env = 'CliffWalking-v0'  # 0 up, 1 right, 2 down, 3 left
        self.result_path = curr_path + "/outputs/" + self.env + '/' + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + '/' + '/models/'  # path to save models
        self.train_eps = 500  # 训练的episode数目
        self.eval_eps = 30
        self.gamma = 0.9  # reward的衰减率
        self.lr = 0.1  # learning rate
        self.render_frqc = 30  # 仿真渲染频率
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env = CliffWalkingWapper(env)
    env.seed(seed)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = QLearning(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    running_rewards = []  # moving average reward
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        state=state[0]
        while True:
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _, _ = env.step(action)  # 与环境进行一次动作交互
            if i_ep % cfg.render_frqc == 0 and i_ep != 0:
                env.render()  # 渲染动作并显示
            agent.update(state, action, reward, next_state,
                         done)  # Q-learning算法更新
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if running_rewards:
            running_rewards.append(running_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            running_rewards.append(ep_reward)

        print("Episode:{}/{}: reward:{:.1f}".format(i_ep + 1, cfg.train_eps,
                                                    ep_reward))
    print('Complete training！')
    return rewards, running_rewards


def eval(cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []  # 记录所有episode的reward
    running_rewards = []  # 滑动平均的reward
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        state = state[0]
        step_count = 0  # 新增：记录每个回合的步数
        while True:
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, _, _ = env.step(action)  # 与环境进行一个交互

            # --- 新增调试打印 ---
            print(
                f"  Eval Episode {i_ep + 1}, Step {step_count}: state={state}, action={action}, reward={reward}, done={done}")
            # --- 结束新增调试打印 ---

            if i_ep % 3 == 0:
                env.render()
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            step_count += 1  # 步数增加

            # 设置一个最大步数，防止无限循环
            if step_count > 200:  # 迷宫最优13步，200步足够发现问题
                print(f"  Eval Episode {i_ep + 1} terminated early due to max steps reached ({step_count} steps).")
                done = True  # 强制结束当前回合

            if done:
                break
        rewards.append(ep_reward)
        if running_rewards:
            running_rewards.append(running_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            running_rewards.append(ep_reward)
        print(f"Episode:{i_ep + 1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")

    print('Complete evaling！')
    return rewards, running_rewards


if __name__ == "__main__":
    cfg = QlearningConfig()  # 获得实验参数
    # 训练智能体
    env = gym.make(cfg.env)
    env = CliffWalkingWapper(env)
    # env.seed(cfg.seed)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = QLearning(state_dim, action_dim, cfg)
    rewards, running_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, running_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards,
                 running_rewards,
                 tag="train",
                 env=cfg.env,
                 algo=cfg.algo,
                 path=cfg.result_path)

    # 测试智能体
    rewards, running_rewards = eval(cfg, env, agent)
    save_results(rewards, running_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards,
                 running_rewards,
                 tag="eval",
                 env=cfg.env,
                 algo=cfg.algo,
                 path=cfg.result_path)
