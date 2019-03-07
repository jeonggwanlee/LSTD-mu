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

TRANSITION = 15000
#EPISODE = 20
#MEMORY_SIZE = TRANSITION + 1000
NUM_ACTION_ITER = 10000
NUM_EVALUATION = 100
NUM_EPISODES = 300
MAX_STEPS = 300
EPOCH_SIZE = 100
BATCH_SIZE = 100


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



class DeepActionNetwork:
    """ Deep Action(Q) Network
        predict action from state

        loss : square(q_pred(s) - q(s, true_a)(=1) )
    """
    def __init__(self,
                 feature_op,
                 state_size=4,
                 action_size=2,
                 n_h1=20,
                 n_h2=9,
                 learning_rate=0.05,
                 scope="deep_action"
                ):
        self.sess = tf.Session()
        self.feature_op = feature_op
        assert self.feature_op in ["h1", "h2", "pred"]
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.scope = scope
        self.meta_name = "./dan_savefile/dan_cartpole_Nh1{}_Nh2{}.meta".format(n_h1, n_h2)
        print("meta_name : {}".format(self.meta_name))
        if self.isRestore():
            self.saver = tf.train.import_meta_graph(self.meta_name)
            self.saver.restore(self.sess, self.meta_name[:-5])
            self._load_network()
        else:
            theta = self._build_network()
            init_new_vars_op = tf.variables_initializer(theta)
            self.sess.run(init_new_vars_op)
            #self.sess.run(tf.global_variable_initializer())

    def _build_network(self):
        with tf.variable_scope(self.scope):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size], name="state_input")
            self.action = tf.placeholder(tf.int32, [None], name="action")
            self.fc1 = tf_utils.fc(self.state_input, self.n_h1, scope="fc1",
                              activation_fn=tf.nn.relu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.fc1_softmax = tf.nn.softmax(self.fc1, name="fc1_softmax")
            self.fc2 = tf_utils.fc(self.fc1, self.n_h2, scope="fc2",
                                   activation_fn=tf.nn.relu,
                                   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.fc2_softmax = tf.nn.softmax(self.fc2, name="fc2_softmax")
            self.q_value = tf_utils.fc(self.fc2, self.action_size, scope="q_value", activation_fn=None)

            self.action_pred = tf.nn.softmax(self.q_value, name="action_prediction")
            self.action_target = tf.one_hot(self.action, self.action_size, on_value=1.0, off_value=0.0,
                    name="action_target")
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_target,
                                                                   logits=self.action_pred, name="loss")
            #self.loss = tf.reduce_mean(tf.square(tf.subtract(self.action_pred, self.action_target)))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),
                    name="train_op")
            new_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return new_variables

    def _load_network(self):
        graph = tf.get_default_graph()
        nodes = graph.as_graph_def().node
        #for node in nodes:
        #    print(node.name)
        #ops = graph.get_operations()
        #for op in ops:
        #    print(op.name)
        self.state_input = graph.get_tensor_by_name("deep_action/state_input:0")
        self.action = graph.get_tensor_by_name("deep_action/action:0")
        self.fc1 = graph.get_tensor_by_name("deep_action/fc1/Relu:0")
        self.fc1_softmax = graph.get_tensor_by_name("deep_action/fc1_softmax:0")
        self.fc2 = graph.get_tensor_by_name("deep_action/fc2/Relu:0")
        self.fc2_softmax = graph.get_tensor_by_name("deep_action/fc2_softmax:0")
        self.q_value = graph.get_tensor_by_name("deep_action/q_value/Add:0")
        self.action_pred = graph.get_tensor_by_name("deep_action/action_prediction:0")
        self.action_target = graph.get_tensor_by_name("deep_action/action_target:0")
        self.loss = graph.get_tensor_by_name("deep_action/loss:0")
        #self.optimizer = graph.get_tensor_by_name("deep_action/optimizer:0")
        self.train_op = graph.get_operation_by_name("deep_action/train_op")

    def isRestore(self):
        #if False:
        if os.path.exists(self.meta_name):
            return True
        else:
            return False

    def _num_basis(self):
        return self.n_h2

    def learn(self, expert_trajectories=None):
        """ training from expert_trajectories """

        if expert_trajectories is None:
            env = gym.make("CartPole-v0")
            expert_trajectories = generate_trajectories_from_expert_policy(env, n_trajectories=100)

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
                if type(loss) == np.float32:
                    print("Deep Action Network Training iteration {}, {}".format(i, loss))
                else:
                    print("Deep Action Network Training iteration {}, {}".format(i, sum(loss)/BATCH_SIZE))

        print("saveing our trained weights!!")
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, "./" + self.meta_name[:-5])

    def get_optimal_action(self, state):
        actions = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        return actions.argmax()

    def get_q_value(self, state):
        q_value = self.sess.run(self.q_value, feed_dict={self.state_input: [state]})
        return q_value

    def get_action_pred(self, state):
        action_pred = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        #q_value = self.sess.run(self.q_value, feed_dict={self.state_input: [state]})
        return action_pred

    def get_features(self, state):
        if self.feature_op == 'pred':
            features = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        elif self.feature_op == 'h2':
            features = self.sess.run(self.fc2_softmax, feed_dict={self.state_input: [state]})
        elif self.feature_op == 'h1':
            features = self.sess.run(self.fc1_softmax, feed_dict={self.state_input: [state]})
        return features

    def get_feature_dim(self):
        if self.feature_op == 'pred':
            return self.action_size
        elif self.feature_op == 'h2':
            return self.n_h2
        elif self.feature_op == 'h1':
            return self.n_h1
 
    def evaluate_multi_states(self, state):
        """ get features's multiple version
        """
        if self.feature_op == 'pred':
            features = self.sess.run(self.action_pred, feed_dict={self.state_input: state})
        elif self.feature_op == 'h2':
            features = self.sess.run(self.fc2_softmax, feed_dict={self.state_input: state})
        elif self.feature_op == 'h1':
            features = self.sess.run(self.fc1_softmax, feed_dict={self.state_input: state})
        return features

    def test(self, env, isRender=True, num_test=100):
        print("Testing Deep Action Network... {} times".format(num_test))
        timesteps = []
        for i in range(num_test):
            cur_state = env.reset()
            done = False
            t = 0
            while not done:
                t = t + 1
                if isRender:
                    env.render()
                action = self.get_optimal_action(cur_state)
                next_state, reward, done, _ = env.step(action)
                cur_state = next_state
                if done:
                    print("Test DAN {} : {} timesteps".format(i, t))
                    timesteps.append(t)
                    break
        print("DAN average test results : {}".format(sum(timesteps)/num_test))
            #end while
        #end for i
                
