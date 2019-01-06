import pickle
import ipdb
import gym
import numpy as np
import copy
import os
import datetime
import tensorflow as tf

from record import get_test_record_title
from replay_memory import Memory
from lspi import LSPI
from lstd_mu import LSTD_MU
from deep_action_network import DeepActionNetwork
from irl_test import IRL_test

TRANSITION = 15000
EPISODE = 30
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000
NUM_EVALUATION = 100

PSI_S0_ITERATION = 1000
NUM_LSTDMU_SAMPLING = 1000

important_sampling = True

class IRL_LSTDMU_DAN:
    def __init__(self, env, dan, expert_trajectories, gamma, epsilon):
        self.env = env
        self.dan = dan
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = None

        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        action_dim = 1
        self.memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, self.state_dim)
        self.mu_expert = self.compute_feature_expectation(expert_trajectories)


    def _generate_trajectories_from_initial_policy(self, n_trajectories=1000):
        trajectories = []
        for _ in range(n_trajectories):
            self.env.seed()
            state = self.env.reset()
            trajectory = []
            for _ in range(TRANSITION):  # TRANSITION
                #env.render()
                if state[0] > 0:         # right
                    action = 0           # go right
                    next_state, reward, done, info = self.env.step(action)
                else: # left
                    action = 1           # go left
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
                phi_state = self.dan.get_features(state)
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

    def _test_policy_with_approxi_reward(self, agent, dan, theta, isRender=False):
        total_reward = 0.0
        state = self.env.reset()
        for _ in range(TRANSITION):
            if isRender:
                self.env.render()
            phi = self.dan.get_features(state)
            approxi_reward = np.dot(phi, theta.T).min()
            action = agent.act(state)
            next_state, _, done, _ = self.env.step(action)
            state = next_state
            total_reward += approxi_reward 
            if done:
                break

        return total_reward


    def _get_best_agent(self, memory, agent, theta, isRender=False):
        Best_agent = None
        Best_mean_reward = float("-inf")
        mean_reward = float("-inf")
        # 1000 * 100
        for i in range(EPISODE):
            self.env.seed()
            state = self.env.reset()
            for j in range(TRANSITION):
                if isRender:
                    self.env.render()
                action = self.env.action_space.sample()
                next_state, _, done, _ = self.env.step(action)
                phi_state = self.dan.get_features(state)
                reward = np.dot(phi_state, theta.T).min()
                memory.add([state, action, reward, next_state, done])
                state = next_state
                if done:
                    break

            if memory.container_size < BATCH_SIZE:
                sample = memory.select_sample(memory.container_size)
            else:
                sample = memory.select_sample(BATCH_SIZE)
            
            error = agent.train(sample, w_important_sampling=important_sampling)

            reward_list = []
            for j in range(NUM_EVALUATION):
                total_reward = self._test_policy_with_approxi_reward(agent,
                                                                     self.dan,
                                                                     theta)
                reward_list.append(total_reward)
            mean_reward = sum(reward_list) / NUM_EVALUATION

            if Best_mean_reward < mean_reward:
                #print("Get Best reward {}".format(mean_reward))
                Best_agent = copy.deepcopy(agent)
                Best_mean_reward = mean_reward
                memory.clear_memory()
            
            if i % 20 == 0:
                print("Find Best Agent iteration : {}/{}".format(i, EPISODE))
            
        # for i
        # Clean up
        memory.clear_memory()

        return Best_agent


    def loop(self, loop_iter):

        pre_soft = self.dan.pre_soft
        dan_output = self.dan._num_basis()
        best_policy_bin_name = "CartPole-v0_DAN{}_PreSoft{}_ImportantSampling{}_FindBestAgentEpi{}_best_policy_irl_lstdmu_pickle_{}.bin".format(dan_output, pre_soft, important_sampling, EPISODE, loop_iter)
        print("#Experiment name : ", best_policy_bin_name)

        iteration = 0
        Best_agents = []
        t_collection = []
        test_reward_collection = []

        # 1.
        initial_trajectories = self._generate_trajectories_from_initial_policy()
        self.mu_initial = self.compute_feature_expectation(initial_trajectories)

        # 2.
        self.mu_bar = self.mu_initial
        self.theta = self.mu_expert - self.mu_bar  # theta
        t = np.linalg.norm(self.theta, 2)
        print("Initial threshold: ", t)

        # 3.
        agent = LSPI(self.num_actions, self.state_dim)
        psi_function = agent.basis_function
        q = psi_function._num_basis()

        while t > self.epsilon:
            print("iteration: ", iteration)
            # 4.
            agent = LSPI(self.num_actions, self.state_dim)
            best_agent = self._get_best_agent(self.memory,
                                              agent,
                                              self.theta,
                                              isRender=False)

            self.memory.clear_memory()
            Best_agents.append(best_agent)

            test_reward = IRL_test(self.env, best_agent, iteration)
            test_reward_collection.append(test_reward)

            # Time checker 1
            # start = datetime.datetime.now()

            # Get psi(s0)
            _psi_sum = np.zeros((q,))
            for i in range(PSI_S0_ITERATION):
                _psi = psi_function.evaluate(self.env.reset(), self.env.action_space.sample())
                _psi_sum += _psi
            psi = _psi_sum / PSI_S0_ITERATION
            psi = np.resize(psi, [len(psi), 1])

            # sampling
            for i in range(NUM_LSTDMU_SAMPLING):
                state = self.env.reset()
                for j in range(TRANSITION):
                    action = best_agent.act(state)
                    next_state, _, done, _ = self.env.step(action)
                    self.memory.add([state, action, None, next_state, done])
                    if done:
                        break

            # collect sample done
            jth_policy = best_agent.policy
            lstd_mu = LSTD_MU(psi_function, self.gamma)
            samples = self.memory.select_sample(BATCH_SIZE * 2)
            self.memory.clear_memory()
            if important_sampling:
                xi = lstd_mu.train_parameter_with_important_sampling(samples,
                                                                     jth_policy,
                                                                     self.dan)
            else:
                xi = lstd_mu.train_parameter(samples, jth_policy, self.dan)
            mu_origin = np.dot(xi.T, psi)
            mu_origin = np.reshape(mu_origin, [-1])
            
            # Time checker 2
            # first_end = datetime.datetime.now()
            # first_end_result = first_end - start

            new_trajectories = self._generate_new_trajectories(best_agent, n_trajectories=1000)
            mu = self.compute_feature_expectation(new_trajectories)
            
            # Time checker 3
            # second_end = datetime.datetime.now()
            # second_end_result = second_end - first_end
            # print(first_end_result)
            # print(second_end_result)

            #updated_loss = mu - self.mu_bar

            updated_loss = mu_origin - self.mu_bar
            self.mu_bar += updated_loss * updated_loss.dot(self.theta.T) / np.square(updated_loss).sum()
            self.theta = self.mu_expert - self.mu_bar
            t = np.linalg.norm(self.theta, 2)
            t_collection.append(t)
            print("threshold: ", t)
            if iteration > 0:
                print("threshold_gap: %05f" % (t_collection[-1] - t_collection[-2]))
            iteration += 1

            if os.path.exists(best_policy_bin_name):
                os.remove(best_policy_bin_name)
            with open(best_policy_bin_name, 'wb') as f:
                pickle.dump([Best_agents, t_collection, test_reward_collection], f)

            if iteration == 200:
                break
        # end while
        
        return


if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 999, 'keepBA&notRB', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_#Trajectories100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        expert_trajectories = pickle.load(rf)  #[[state, action, reward, next_state, done], ...]

    env = gym.make("CartPole-v0")
    sess = tf.Session()
    dan = DeepActionNetwork(sess, pre_soft=False)
    dan.learn(expert_trajectories)

    gamma = 0.99
    epsilon = 0.1

    loop_iteration = list(range(10))[2:]
    for it in loop_iteration:
        irl = IRL_LSTDMU_DAN(env, dan, expert_trajectories, gamma, epsilon)
        irl.loop(it)

