""" Jeonggwan Lee(leejk526@kaist.ac.kr)
"""
import pickle
import ipdb
import gym
import tensorflow as tf

from record import get_test_record_title
from deep_action_network import DeepActionNetwork
from deep_reward_network import DeepRewardNetwork
#from deep_q_network import DQN
from deep_q_network_without_drn import DQN

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
    exp_name = get_test_record_title("CartPole-v0", 999, 'keepBA&notRB', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_#Trajectories100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        expert_trajectories = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    sess = tf.Session()
    env = gym.make("CartPole-v0")

    dan = DeepActionNetwork(sess, pre_soft=True)
    dan.learn(expert_trajectories)
    dan.test(env, isRender=False, num_test=1000)
    
    drn = DeepRewardNetwork(sess, dan)
    drn.learn(env)

    dqn = DQN(sess, drn)
    dqn.learn(env)
