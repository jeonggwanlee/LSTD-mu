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

TRANSITION = 15000


def generate_trajectories_from_expert_policy(env, n_trajectories=100):
    trajectories = []
    rewards_list = []
    for _ in range(n_trajectories):
        state = env.reset()
        trajectory = []
        rewards = 0
        for _ in range(TRANSITION):
            if state[2] < 0:        # pole angle is minus(left)
                if state[3] < 0:    # pole velocity is minus(left) => bad situation.
                    action = 0      # go left
                else:               # pole velocity is plus(right) => good situation.
                    action = env.action_space.sample()
            else:                   # pole angle is plus(right)
                if state[3] < 0:    # pole velocity is minus(left) => good situation.
                    action = env.action_space.sample()
                else:
                    action = 1      # go right
            next_state, reward, done, info = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            state = next_state
            rewards += 1
            if done:
                rewards_list.append(rewards)
                break
        # for j
        trajectories.append(trajectory)
    # for i
    print("expert policy average reward : {}".format(sum(rewards_list)/n_trajectories))
    return trajectories



if __name__ == '__main__':

    sess = tf.Session()
    env = gym.make("CartPole-v0")

    expert_trajectories = generate_trajectories_from_expert_policy(env, n_trajectories=300)

    dan = DeepActionNetwork("h2")
    if not dan.isRestore():
        dan.learn(expert_trajectories)
    dan.test(env, isRender=False, num_test=100)
    
#    drn = DeepRewardNetwork(sess, dan)
#    drn.learn(env)

    dqn = DQN(sess, dan)
    dqn.learn(env)
