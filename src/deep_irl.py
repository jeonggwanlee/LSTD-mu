""" Jeonggwan Lee(leejk526@kaist.ac.kr)
"""
import pickle
import ipdb
import gym
import numpy as np
import tensorflow as tf
import random

from record import get_test_record_title
from deep_action_network import DeepActionNetwork
from deep_reward_network import DeepRewardNetwork
from deep_q_network import DQN
import tf_utils

# TRANSITION = 15000
# EPISODE = 20
# MEMORY_SIZE = TRANSITION + 1000
NUM_ACTION_ITER = 1000
NUM_REWARD_ITER = 1000
NUM_EVALUATION = 100
NUM_EPISODES = 300
MAX_STEPS = 300
EPOCH_SIZE = 100
BATCH_SIZE = 100
START_MEM = 100




if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 1000, 'initial2', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_num_traj100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        expert_trajectories = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    sess = tf.Session()
    env = gym.make("CartPole-v0")

    dan = DeepActionNetwork(sess)
    dan.learn(expert_trajectories)
    """
    env = gym.make("CartPole-v0")
    while True:
        cur_state = env.reset()
        done = False
        t = 0
        while not done:
            env.render()
            t = t+1
            action = dan.get_optimal_action(cur_state)
            print(action)
            next_state, reward, done, info = env.step(action)
            cur_state = next_state
            if done:
                print("{} timesteps".format(t+1))
                break
    """
    drn = DeepRewardNetwork(sess, dan)
    drn.learn(env)

    dqn = DQN(sess, drn)
    dqn.learn(env)
