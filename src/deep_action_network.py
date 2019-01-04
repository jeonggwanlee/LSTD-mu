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


class DeepActionNetwork:
    """ Deep Action(Q) Network
        predict action from state

        loss : square(q_pred(s) - q(s, true_a)(=1) )
    """
    def __init__(self,
                 session,
                 learning_rate=0.05,
                 state_size=4,
                 action_size=2,
                 n_h1=20,
                 n_h2=20,
                 scope="deep_action"
                ):
        self.sess = session
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.scope = scope
        theta = self._build_network()
        init_new_vars_op = tf.variables_initializer(theta)
        self.sess.run(init_new_vars_op)

    def _build_network(self):
        with tf.variable_scope(self.scope):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
            self.action = tf.placeholder(tf.int32, [None])
            self.fc1 = tf_utils.fc(self.state_input, self.n_h1, scope="fc1",
                              activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.fc2 = tf_utils.fc(self.fc1, self.n_h2, scope="fc2",
                                   activation_fn=tf.nn.elu,
                                   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.fc2_softmax = tf.nn.softmax(self.fc2, name="fc2_softmax")
            self.q_value = tf_utils.fc(self.fc2, self.action_size, activation_fn=None)
            self.action_pred = tf.nn.softmax(self.q_value, name="action")
            self.action_target = tf.one_hot(self.action, self.action_size, on_value=1.0, off_value=0.0)
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.action_pred, self.action_target)))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            new_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return new_variables

    def learn(self, expert_trajectories):
        """ training from expert_trajectories """
        expert_trajs_flat = []
        for i in range(len(expert_trajectories)):
            for j in range(len(expert_trajectories[i])):
                expert_trajs_flat.append(expert_trajectories[i][j])
        random.shuffle(expert_trajs_flat)

        batch_end = 0
        for i in range(NUM_ACTION_ITER):
            if batch_end + BATCH_SIZE > len(expert_trajs_flat):
                batch_end = 0
                random.shuffle(expert_trajs_flat)
            batch_expert_trajs = expert_trajs_flat[batch_end:batch_end+BATCH_SIZE]
            cur_state_batch = [s[0] for s in batch_expert_trajs]
            cur_action_batch = [s[1] for s in batch_expert_trajs]
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.state_input:cur_state_batch,
                                                                           self.action:cur_action_batch})
            batch_end += BATCH_SIZE
            if i % 10 == 0:
                print("i {}, {}".format(i, loss))

    def get_optimal_action(self, state):
        actions = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        return actions.argmax()

    def get_q_value(self, state):
        q_value = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        return q_value

    def get_features(self, state):
        fc2 = self.sess.run(self.fc2_softmax, feed_dict={self.state_input: [state]})
        return fc2
