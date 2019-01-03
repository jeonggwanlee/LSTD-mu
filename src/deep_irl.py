"""
"""
import pickle
import ipdb
import gym
import numpy as np
import copy
import os
import tensorflow as tf
import random

from reward_basis import RewardBasis, Theta
from record import get_test_record_title
from replay_memory import Memory
from lspi_for_ap import LSPI_AP
import tf_utils

TRANSITION = 15000
EPISODE = 20
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000
NUM_EVALUATION = 100


class DeepActionNetwork:
    def __init__(self,
            session,
            epsilon=0.5,
            epsilon_anneal=0.01,
            end_epsilon=0.1,
            lr=0.05,
            gamma=0.99,
            state_size=4,
            action_size=2,
            n_h1=20,
            n_h2=20,
            scope="deep_action"
            ):
        self.sess = session
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.lr = lr
        self.gamma = gamma
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
            fc1 = tf_utils.fc(self.state_input, self.n_h1, scope="fc1",
                    activation_fn=tf.nn.elu,
                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2",
                    activation_fn=tf.nn.elu,
                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.q_value = tf_utils.fc(fc2, self.action_size, activation_fn=None)

            self.action_pred = tf.nn.softmax(self.q_value, name="action")
            self.action_target = tf.one_hot(self.action, self.action_size, on_value=1.0, off_value=0.0)
        
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.action_pred, self.action_target)))
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

        return theta

    def learn(self, expert_trajectories):
        batch_size = 100
        expert_trajs_flat = []
        for i in range(len(expert_trajectories)):
            for j in range(len(expert_trajectories[i])):
                expert_trajs_flat.append(expert_trajectories[i][j])
        
        random.shuffle(expert_trajs_flat)
        
        batch_end = 0
        
        for i in range(1000):
            if batch_end + batch_size > len(expert_trajs_flat):
                batch_end = 0
                random.shuffle(expert_trajs_flat)
            batch_expert_trajs = expert_trajs_flat[batch_end:batch_end+batch_size]
            cur_state_batch = [s[0] for s in batch_expert_trajs]
            cur_action_batch = [s[1] for s in batch_expert_trajs]
            l, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.state_input:cur_state_batch,
                                                                        self.action:cur_action_batch})
            batch_end += batch_size
            if i % 10 == 0:
                print("i {}, {}".format(i, l))

    def get_optimal_action(self, state):
        actions = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        return actions.argmax()

   
    def get_q_value(self, state):
        q_value = self.sess.run(self.action_pred, feed_dict={self.state_input: [state]})
        return q_value


class DeepRewardNetwork:
    def __init__(self,
            session,
            qnet,
            epsilon=0.5,
            epsilon_anneal=0.01,
            end_epsilon=0.1,
            lr=0.005,
            gamma=0.9,
            state_size=4,
            action_size=2,
            n_h1=20,
            n_h2=20,
            scope="deep_reward"
            ):
        self.sess = session
        self.qnet = qnet
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.lr = lr
        self.gamma = gamma
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
            self.reward_target = tf.placeholder(tf.float32, [None])
            fc1 = tf_utils.fc(self.state_input, self.n_h1, scope="fc1",
                    activation_fn=tf.nn.elu,
                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2",
                    activation_fn=tf.nn.elu,
                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            self.reward_pred = tf_utils.fc(fc2, 1, activation_fn=None)

            self.loss = tf.nn.l2_loss(tf.subtract(self.reward_pred, self.reward_target))
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return theta

    
    def learn(self, env):
        trajectories = []
        for i in range(1000):
            state = env.reset()
            for j in range(2000):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                trajectories.append([state, action, next_state])
                state = next_state
                if done:
                    break

        random.shuffle(trajectories)
        batch_size = 100
        batch_end = 0
        
        for i in range(1000):
            if batch_end + batch_size > len(trajectories):
                batch_end = 0
                random.shuffle(trajectories)
            batch_trajectories = trajectories[batch_end:batch_end+batch_size]

            batch_states = [traj[0] for traj in batch_trajectories]
            #values = [self.qnet.get_q_value(traj[0]).reshape([-1]) for traj in batch_trajectories]
            #q_values = [values[i][traj[1]] for i, traj in enumerate(batch_trajectories)]
            q_values = [self.qnet.get_q_value(traj[0])[0, traj[1]] for traj in batch_trajectories]

            next_max_q_values = [np.max(self.qnet.get_q_value(traj[2])) for traj in batch_trajectories]

            q_values = np.asarray(q_values)
            next_max_q_values = np.asarray(next_max_q_values)
            reward_targets = q_values - self.gamma * next_max_q_values
            l, _ = sess.run([self.loss, self.train_op], feed_dict={self.state_input:batch_states,
                                                            self.reward_target:reward_targets})
            
            if i % 10 == 0:
                print("i : {}, {}".format(i, l))
                """
                if i > 100:
                    gaps = []
                    gap_iter = 0
                    reward = None
                    reward_before = None
                    reward_pred_before = None
                    reward_pred = None
                    for i in range(1):
                        state = env.reset()
                        for j in range(2000):
                            gap_iter += 1
                            action = env.action_space.sample()
                            reward_before = reward
                            next_state, reward, done, info = env.step(action)
                            reward_pred_before = reward_pred
                            reward_pred = sess.run(self.reward_pred, feed_dict={self.state_input:[state]})
                            gaps.append(reward_pred[0][0] - reward)
                            if done:
                                print(j)
                                print("now ", reward_pred[0][0])
                                print("before ", reward_pred_before[0][0])
                                print("reward true ", reward)
                                print("reward true_before ", reward_before)
                                break
                    gap = sum(gaps) / gap_iter
                    print("gap : {}\n".format(gap))
                """
        # for end
        ipdb.set_trace()


    def get_reward(self, state):
        reward = self.sess.run(self.reward_pred, feed_dict={self.state_input:[state]})
        return reward[0][0]


class DQN:
    def __init__(self,
            session,
            epsilon=0.5,
            epsilon_anneal=0.01,
            end_epsilon=0.1,
            lr=0.005,
            gamma=0.9,
            state_size=4,
            action_size=2,
            n_h1=20,
            n_h2=20,
            scope="dqn"
            ):
        self.sess = session
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.lr = lr
        self.gamma = gamma
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
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return theta

    def learn(self):




class DEEPIRL:
    def __init__(self, env, reward_basis, expert_trajectories, gamma, epsilon,
            n_input, lr, n_h1, n_h2, name='deep_irl'):
        self.env = env
        self.reward_basis = reward_basis
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = None

        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        action_dim = 1
        self.memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim) 

        self.mu_expert = self.compute_feature_expectation(expert_trajectories)

        self.n_input = n_input
        self.lr = lr
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.name = name

        self.sess = tf.Session()


    def _generate_trajectories_from_initial_policy(self, n_trajectories=1000):
        trajectories = []
        for _ in range(n_trajectories):
            self.env.seed()
            state = self.env.reset()
            trajectory = []
            for _ in range(TRANSITION): # TRANSITION
                #env.render()
                if state[0] > 0: # right
                    action = 0 # go right
                    next_state, reward, done, info = self.env.step(action)
                else: # left
                    action = 1 # go left
                    next_state, reward, done, info = self.env.step(action)
                trajectory.append([state, action, reward, next_state, done])
                state = next_state
                if done:
                    break
            # for j
            trajectories.append(trajectory)
        # for i
        return trajectories

    def _generate_new_trajectories(self, agent, n_trajectories=1000):
        trajectories = []
        for _ in range(n_trajectories):
            self.env.seed()
            state = self.env.reset()

            trajectory = []
            for _ in range(TRANSITION):
                action = agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append([state, action, reward, next_state, done])
                state = next_state
                if done:
                    break
            # for _
            trajectories.append(trajectory)
        # for _
        return trajectories


    def compute_feature_expectation(self, trajectories):
        mu_sum = None
        for i, one_traj in enumerate(trajectories):

            one_mu = None
            gamma_update = 1.0 / self.gamma
            for j, sample in enumerate(one_traj): # [s, a, r, s', d]
                state = sample[0]
                phi_state = self.reward_basis.evaluate(state)
                gamma_update *= self.gamma
                phi_time_unit = phi_state * gamma_update
                if j == 0: 
                    one_mu = phi_time_unit
                else: 
                    one_mu += phi_time_unit
            # for j
            if i == 0: 
                mu_sum = one_mu
            else: 
                mu_sum += one_mu
        # for i
        mu = mu_sum / len(trajectories)
        return mu

    def _test_policy_with_approxi_reward(self, agent, reward_basis, theta, isRender=False):
        total_reward = 0.0
        self.env.seed()
        state = self.env.reset()
        for _ in range(TRANSITION):
            if isRender:
                self.env.render()
            phi = reward_basis.evaluate(state)
            approxi_reward = np.dot(phi, theta)
            action = agent.act(state)
            next_state, _, done, _ = self.env.step(action)
            state = next_state
            total_reward += approxi_reward 
            if done:
                break

        return total_reward


    def _get_best_agent(self, memory, agent, theta, isRender=False):
        Best_agent = None
        Best_mean_reward = -4000
        mean_reward = -4000
        # 1000 * 100
        for i in range(EPISODE):
            self.env.seed()
            state = self.env.reset()
            for j in range(TRANSITION):
                if isRender:
                    self.env.render()
                action = self.env.action_space.sample()
                next_state, _, done, _ = self.env.step(action)
                phi_state = self.reward_basis.evaluate(state)
                reward = np.dot(phi_state, theta)
                memory.add([state, action, reward, next_state, done])
                state = next_state
                if done:
                    break

            if memory.container_size < BATCH_SIZE:
                sample = memory.select_sample(memory.container_size)
            else:
                sample = memory.select_sample(BATCH_SIZE)
            
            agent.train(sample, w_important_sampling=True)
            
            reward_list = []
            for j in range(NUM_EVALUATION):
                total_reward = self._test_policy_with_approxi_reward(agent, self.reward_basis,
                                                                      theta)
                reward_list.append(total_reward)
            mean_reward = sum(reward_list) / NUM_EVALUATION

            if Best_mean_reward < mean_reward:
                print("Get Best reward {}".format(mean_reward))
                Best_agent = copy.deepcopy(agent)
                Best_mean_reward = mean_reward
                memory.clear_memory()
            
            if i % 20 == 0:
                print("i : {}/{}".format(i, EPISODE))
            
        # for i
        # Clean up
        memory.clear_memory()

        return Best_agent


    def loop(self):

        # 1.
        initial_trajectories = self._generate_trajectories_from_initial_policy()
        self.mu_initial = self.compute_feature_expectation(initial_trajectories)

        # 2.
        self.mu_bar = self.mu_initial
        self.theta = self.mu_expert - self.mu_bar # theta
        t = np.linalg.norm(self.theta, 2)
        print("Initial threshold: ", t)
        iteration = 0
        Best_agents = []
        t_collection = []
        best_policy_bin_name = "CartPole-v0_statedim4_numbasis10_best_policy_pickle.bin"
        # R = pi w
        # 3.
        while t > self.epsilon:
            # 4.
            agent = LSPI_AP(self.num_actions, self.state_dim)
            best_agent = self._get_best_agent(self.memory, agent, 
                                              self.theta, isRender=False)
            Best_agents.append(best_agent)

            new_trajectories = self._generate_new_trajectories(best_agent, n_trajectories=1000)
            mu = self.compute_feature_expectation(new_trajectories)            
            updated_loss = mu - self.mu_bar
            self.mu_bar += updated_loss * updated_loss.dot(self.theta) / np.square(updated_loss).sum()
            self.theta = self.mu_expert - self.mu_bar
            t = np.linalg.norm(self.theta, 2)
            t_collection.append(t)
            print("iteration: ", iteration)
            print("threshold: ", t)
            if iteration > 1:
                print("threshold_gap: ", t_collection[-1] - t_collection[-2])
            iteration += 1

            if os.path.exists(best_policy_bin_name):
                os.remove(best_policy_bin_name)
            with open(best_policy_bin_name, 'wb') as f:
                pickle.dump([Best_agents, t_collection], f)

        
        return


if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 1000, 'initial2', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_num_traj100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        expert_trajectories = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    sess = tf.Session()
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
    env = gym.make("CartPole-v0")

    drn.learn(env)


