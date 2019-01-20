""" Jeonggwan Lee(leejk526@kaist.ac.kr)
"""
import pickle
import ipdb
import gym
import numpy as np
import tensorflow as tf
import random

from record import get_test_record_title
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


class DQN:
    """ Deep Q Network
        predict

        \hat(q(s, a)) - q(s, a)
        q(s, a) = reward - gamma * max_{a`}q(s`, a`)
    """
    def __init__(self,
                 session,
                 qnet,
                 epsilon=1,
                 epsilon_anneal=0.01,
                 end_epsilon=0.1,
                 learning_rate=0.001,
                 gamma=0.9,
                 state_size=4,
                 action_size=2,
                 n_h1=20,
                 n_h2=20,
                 scope="dqn"
                ):
        self.sess = session
        self.qnet = qnet
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.scope = scope
        new_variables = self._build_network()
        self.sess.run(tf.variables_initializer(new_variables))
        self.memory = []
        self.memory_size = 0
        self.memory_limit = 10000

    def _build_network(self):
        with tf.variable_scope(self.scope):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
            self.action = tf.placeholder(tf.int32, [None])
            self.target_q = tf.placeholder(tf.float32, [None])
            fc1 = tf_utils.fc(self.state_input, self.n_h1, scope="fc1",
                              activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2",
                              activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.q_values = tf_utils.fc(fc2, self.action_size, activation_fn=None)

            action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)

            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, q_value_pred)))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            variables_in_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return variables_in_scope

    def memory_add(self, state, action, next_state, reward, done):
        if self.memory_size + 1 >= self.memory_limit:
            self.memory = self.memory[1:]
            self.memory_size -= 1
        self.memory.append([state, action, next_state, reward, done])
        self.memory_size += 1

    def learn(self, env):
        for i_epi in range(NUM_EPISODES):
            state = env.reset()
            average_r = []
            for t in range(MAX_STEPS):
                #env.render()
                action = self.get_action(state)
                next_state, _, done, _ = env.step(action)

                q_value = self.qnet.get_q_value(state)[0, action]
                next_max_q_value = np.max(self.qnet.get_q_value(next_state))
                reward = q_value - self.gamma * next_max_q_value
                #reward = self.rnet.get_reward(state) + 1
                #print("DQN Training episode {} timestep {} reward {}".format(i_epi,
                #                                                             t,
                #                                                             reward))
                #reward = self.rnet.get_reward(state)
                average_r.append(reward)
                # reward = 1
                if done:
                    reward = -1 # rnet mean is -0.4
                    self.memory_add(state, action, next_state, reward, done)
                    print("DQN Training Episode {} finished after {} timesteps".format(i_epi, t+1))
                    print("average reward : {}".format(sum(average_r)/t))
                    break
                self.memory_add(state, action, next_state, reward, done)
                state = next_state
            self.epsilon_decay()

            for epoch in range(EPOCH_SIZE):
                #if self.memory_size < BATCH_SIZE:
                #    batch = random.sample(self.memory, self.memory_size)
                #else:
                #    batch = random.sample(self.memory, BATCH_SIZE)
                if self.memory_size < START_MEM:
                    print("not enough!")
                    break
                batch = random.sample(self.memory, BATCH_SIZE)
                # state action next_state reward, done
                next_state_batch = [bat[2] for bat in batch]
                q_values = self.sess.run(self.q_values, feed_dict={self.state_input:
                                                                   next_state_batch})
                max_q_values = q_values.max(axis=1)

                cur_state_batch = [bat[0] for bat in batch]
                action_batch = [bat[1] for bat in batch]
                reward_batch = [bat[3] for bat in batch]
                target_q = np.array([reward_batch[k] + self.gamma*max_q_values[k]*(1-bat[4])
                                                            for k, bat in enumerate(batch)])
                target_q = target_q.reshape([len(batch)])
                # minimize the TD-error
                loss, _ = self.sess.run([self.loss, self.train_op], 
                                        feed_dict={self.state_input: cur_state_batch,
                                                   self.target_q: target_q,
                                                   self.action: action_batch})
            # for epoch
        # for i
        ipdb.set_trace()

        while True:
            cur_state = env.reset()
            done = False
            t = 0
            while not done:
                env.render()
                t = t + 1
                action = self.get_optimal_action(cur_state)
                next_state, reward, done, info = env.step(action)
                cur_state = next_state
                if done:
                    print("{} timesteps".format(t+1))
                    break


    def get_optimal_action(self, state):
        actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
        return actions.argmax()

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return self.get_optimal_action(state)

    def epsilon_decay(self):
        if self.epsilon > self.end_epsilon:
            self.epsilon = self.epsilon - self.epsilon_anneal


