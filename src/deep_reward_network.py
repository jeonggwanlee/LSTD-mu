""" Jeonggwan Lee(leejk526@kaist.ac.kr)
"""
import pickle
import ipdb
import gym
import numpy as np
import copy
import os
import tensorflow as tf
import random

from record import get_test_record_title
import tf_utils

#TRANSITION = 15000
#EPISODE = 20
#MEMORY_SIZE = TRANSITION + 1000
NUM_ACTION_ITER = 1000
NUM_REWARD_ITER = 1000
NUM_EVALUATION = 100
NUM_EPISODES = 300
MAX_STEPS = 300
EPOCH_SIZE = 100
BATCH_SIZE = 100
START_MEM = 100

class DeepRewardNetwork:
    def __init__(self,
                 session,
                 qnet,
                 learning_rate=0.005,
                 gamma=0.9,
                 state_size=4,
                 action_size=2,
                 n_h1=20,
                 n_h2=20,
                 scope="deep_reward"
                ):
        self.sess = session
        self.qnet = qnet
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.scope = scope
        new_variables = self._build_network()
        self.sess.run(tf.variables_initializer(new_variables))

    def _build_network(self):
        with tf.variable_scope(self.scope):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
            self.reward_target = tf.placeholder(tf.float32, [None])
            fc1 = tf_utils.fc(self.state_input, self.n_h1, scope="fc1",
                    activation_fn=tf.nn.elu,
                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2",
                    activation_fn=tf.nn.elu,
                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.reward_pred = tf_utils.fc(fc2, 1, activation_fn=None)

            self.loss = tf.norm(tf.subtract(self.reward_pred, self.reward_target), 2)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return theta

    
    def learn(self, env):
        trajectories = [] # state, action, next_state
        for i in range(NUM_REWARD_ITER):
            state = env.reset()
            for j in range(2000):
                action = env.action_space.sample()
                next_state, _, done, info = env.step(action)
                trajectories.append([state, action, next_state])
                state = next_state
                if done:
                    break

        random.shuffle(trajectories)

        batch_end = 0
        for i in range(NUM_REWARD_ITER):
            if batch_end + BATCH_SIZE > len(trajectories):
                batch_end = 0
                random.shuffle(trajectories)
            batch = trajectories[batch_end:batch_end+BATCH_SIZE]
            cur_states_batch = [bat[0] for bat in batch]
            q_values = [self.qnet.get_q_value(bat[0])[0, bat[1]] for bat in batch]
            q_values = np.asarray(q_values)
            next_max_q_values = [np.max(self.qnet.get_q_value(bat[2])) for bat in batch]
            next_max_q_values = np.asarray(next_max_q_values)

            reward_targets = q_values - self.gamma * next_max_q_values
            loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.state_input:cur_states_batch,
                                                                      self.reward_target:reward_targets})
            
            if i % 10 == 0:
                print("i : {}, {}".format(i, loss))
        # for end

    def get_reward(self, state):
        reward = self.sess.run(self.reward_pred, feed_dict={self.state_input:[state]})
        return reward.reshape([-1])[0]

    def get_rewards(self, states):
        reward = self.sess.run(self.reward_pred, feed_dict={self.state_input:states})
        return reward.reshape([-1])


