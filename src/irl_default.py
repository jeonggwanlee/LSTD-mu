import pickle
import ipdb
import gym
import numpy as np
import copy
import os
# import tensorflow as tf

from reward_basis import RewardBasis
from record import get_test_record_title
from replay_memory import Memory
from lspi import LSPI
# from deep_action_network import DeepActionNetwork
from irl_test import IRL_test

TRANSITION = 15000
EPISODE = 30
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000
NUM_EVALUATION = 100
important_sampling = True

class IRL:
    def __init__(self, env, reward_basis, expert_trajectories, gamma, epsilon):
        self.env = env
        self.reward_basis = reward_basis
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = None

        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        action_dim = 1
        self.memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, self.state_dim)

        self.mu_expert = self.compute_feature_expectation(expert_trajectories)
        initial_trajectories = self._generate_trajectories_from_initial_policy()
        self.mu_initial = self.compute_feature_expectation(initial_trajectories)
        self.mu_bar = self.mu_initial

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
        Best_mean_reward = float("-inf")
        mean_reward = float("-inf")
        # 1000 * 100
        for i in range(EPISODE):
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
            
            agent.train(sample, w_important_sampling=important_sampling)
            
            reward_list = []
            for j in range(NUM_EVALUATION):
                total_reward = self._test_policy_with_approxi_reward(agent,
                                                                     self.reward_basis,
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

        p = self.reward_basis._num_basis()
        best_policy_bin_name = "CartPole-v0_RewardBasis{}_ImportantSampling{}_FindBestAgentEpi{}_best_policy_irl_pickle_{}.bin".format(p, important_sampling, EPISODE, loop_iter)

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
        self.theta = self.mu_expert - self.mu_bar # theta
        t = np.linalg.norm(self.theta, 2)
        print("Initial threshold: ", t)

        # 3.
        while t > self.epsilon:

            print("iteration: ", iteration)
            # 4.
            agent = LSPI(self.num_actions, self.state_dim)
            best_agent = self._get_best_agent(self.memory,
                                              agent,
                                              self.theta,
                                              isRender=False)
            Best_agents.append(best_agent)

            # Best agent testing
            test_reward = IRL_test(self.env, best_agent, iteration)
            test_reward_collection.append(test_reward)

            # 2. Projection method
            new_trajectories = self._generate_new_trajectories(best_agent, n_trajectories=1000)
            mu = self.compute_feature_expectation(new_trajectories)
            updated_loss = mu - self.mu_bar
            self.mu_bar += updated_loss * updated_loss.dot(self.theta) / np.square(updated_loss).sum()
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

        return


if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 999, 'keepBA&notRB', num_tests=1, important_sampling=True)

    num_traj = 100
    
    traj_name = exp_name + '_#Trajectories{}_pickle.bin'.format(num_traj)
    print("trajectory file name {} ".format(traj_name))
    with open(traj_name, 'rb') as rf:
        expert_trajectories = pickle.load(rf)  #[[state, action, reward, next_state, done], ...]

    state_dim = 4
    num_basis = 9
    feature_means_name = "CartPole-v0_RewardBasis{}_pickle.bin".format(num_basis)
    with open(feature_means_name, 'rb') as rf:
        feature_means = pickle.load(rf)

    env = gym.make("CartPole-v0")
    gamma = 0.99
    reward_basis = RewardBasis(state_dim, num_basis, gamma, feature_means)
    epsilon = 0.1

    #iteration_loop = list(range(10))[2:]
    iteration_loop = ["#Trajs{}".format(num_traj)]
    for it in iteration_loop:
        irl = IRL(env, reward_basis, expert_trajectories, gamma, epsilon)
        irl.loop(it)

