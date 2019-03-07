import pickle
import ipdb
import gym
import numpy as np
import copy
import os
# import tensorflow as tf

from reward_basis import RewardBasis
from replay_memory import Memory
from lspi import LSPI
from irl_test import IRL_test

TRANSITION = 15000
EPISODE_FOR_TRAIN = 100
MEMORY_SIZE = 10000
important_sampling = False


class IRL:

    def __init__(self, env, reward_basis, gamma, epsilon, debug_name, lspi_bfdim=20, lspi_bfopt="deep_cartpole",
            num_traj_for_policy=100, num_expert=200, num_traj_for_mu=200, num_eval=100):
        self.env = env
        self.reward_basis = reward_basis
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = None

        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        action_dim = 1
        self.memory = Memory(MEMORY_SIZE, action_dim, self.state_dim)
        self.agent = LSPI(self.num_actions, self.state_dim, lspi_bfdim, gamma=0.99,
                          opt=lspi_bfopt, saved_basis_use=True)

        self.expert_trajectories = self._generate_trajectories_from_expert_policy(
                                                            n_trajectories=num_expert)
        self.mu_expert = self.compute_feature_expectation(self.expert_trajectories)

        self.num_traj_for_policy = num_traj_for_policy
        self.num_traj_for_mu = num_traj_for_mu
        self.num_eval = num_eval

        # Bin name definition
        p = self.reward_basis._num_basis()
        self.csv_name = "CartPole-v0_#Expert{}_#TrajMu{}_#TrajPolicy{}_RBOpt{}_RBDim{}_LSPIBFOpt{}_LSPIBFDim{}_#Eval{}_{}.csv".format(
                num_expert,
                self.num_traj_for_mu,
                num_traj_for_policy,
                self.reward_basis.opt,
                p,
                lspi_bfopt,
                lspi_bfdim,
                num_eval,
                debug_name)
        print("#csv name : ", self.csv_name)
        with open(self.csv_name, 'a') as f:
            f.write("t,best,mean,worst,sd,mean_ar,std_ar\n")


    def _generate_trajectories_from_initial_policy(self, n_trajectories=100):
        trajectories = []
        rewards_list = []
        for _ in range(n_trajectories):
            state = self.env.reset()
            trajectory = []
            rewards = 0
            for _ in range(TRANSITION):  # TRANSITION
                if state[0] > 0:         # right
                    action = 0           # go right
                    next_state, reward, done, info = self.env.step(action)
                else:                    # left
                    action = 1           # go left
                    next_state, reward, done, info = self.env.step(action)
                trajectory.append([state, action, reward, next_state, done])
                state = next_state
                rewards += 1
                if done:
                    rewards_list.append(rewards)
                    break
            # for j
            trajectories.append(trajectory)
        # for i
        print("initial policy0 average reward : {}".format(sum(rewards_list)/n_trajectories))
        return trajectories

    def _generate_trajectories_from_expert_policy(self, n_trajectories=100):
        trajectories = []
        rewards_list = []
        for _ in range(n_trajectories):
            state = self.env.reset()
            trajectory = []
            rewards = 0
            for _ in range(TRANSITION):
                if state[2] < 0:        # pole angle is minus(left)
                    if state[3] < 0:    # pole velocity is minus(left) => bad situation.
                        action = 0      # go left
                    else:               # pole velocity is plus(right) => good situation.
                        action = self.env.action_space.sample()
                else:                   # pole angle is plus(right)
                    if state[3] < 0:    # pole velocity is minus(left) => good situation.
                        action = self.env.action_space.sample()
                    else:
                        action = 1      # go right
                next_state, reward, done, info = self.env.step(action)
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

    def _generate_new_trajectories(self, agent, n_trajectories=1000):
        trajectories = []
        for i in range(n_trajectories):
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
            for j, sample in enumerate(one_traj):  # [s, a, r, s', d]
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

    def agent_train_wrapper(self, memory, theta, isRender=False):

        approx_rewards = []
        for i in range(self.num_traj_for_policy):
            state = self.env.reset()
            for j in range(TRANSITION):
                if isRender:
                    self.env.render()
                action = self.env.action_space.sample()
                next_state, _, done, _ = self.env.step(action)
                #ipdb.set_trace()
                phi_state = self.reward_basis.evaluate(state)
                reward = np.dot(phi_state, theta)
                approx_rewards.append(reward)
                #if i == 0 and j == 0:
                #    print("{}, {} approx reward : {}".format(i, j, reward))
                memory.add([state, action, reward, next_state, done])
                state = next_state
                if done:
                    break
        
            #if i % 20 == 0:
            #    print("agent_train_wrapper i : {}/{}".format(i, EPISODE_FOR_TRAIN))

        mean = sum(approx_rewards) / len(approx_rewards)
        std = np.var(approx_rewards) ** (1/2)
        print("approx reward : {}. approx std : {}".format(mean, std))

        #print("memory.container_size : {}".format(memory.container_size))
        sample = memory.select_sample(memory.container_size)
        self.agent.train(sample, w_important_sampling=important_sampling)

        # Clean up
        memory.clear_memory()
        return mean, std

    def loop(self):

        # Initialization & list definition
        iteration = 0
        t_collection = []
        test_reward_collection = []

        # 1.
        initial_trajectories = self._generate_trajectories_from_initial_policy(n_trajectories=200)
        self.mu_initial = self.compute_feature_expectation(initial_trajectories)

        # 2.
        self.mu_bar = self.mu_initial
        self.theta = self.mu_expert - self.mu_bar  # theta
        t = np.linalg.norm(self.theta, 2)
        print("Initial threshold: ", t)
        initial_t = copy.deepcopy(t)

        # 3.
        while t > self.epsilon:
            # 4.
            self.agent.initialize_policy()
            mean_ar, std_ar = self.agent_train_wrapper(self.memory,
                                     self.theta,
                                     isRender=False)

            # agent testing
            mean, best, worst, variance = IRL_test(self.env, self.agent, iteration, num_eval=self.num_eval)
            test_reward_collection.append(mean)

            # 2. Projection method
            new_trajectories = self._generate_new_trajectories(self.agent,
                                                               n_trajectories=self.num_traj_for_mu)
            mu = self.compute_feature_expectation(new_trajectories)
            updated_loss = mu - self.mu_bar
            self.mu_bar += updated_loss * updated_loss.dot(self.theta) / np.square(updated_loss).sum()
            self.theta = self.mu_expert - self.mu_bar
            t = np.linalg.norm(self.theta, 2)
            t_collection.append(t)
            #print("threshold: ", t)
            if iteration == 0:
                th_gap = initial_t - t_collection[-1]
                print("iteration {0:} threshold : {1:.5f}".format(iteration, t))
            elif iteration > 0:
                th_gap = t_collection[-1] - t_collection[-2]
                print("iteration {0:} threshold : {1:.5f}, threshold_gap: {2:.5f}".format(
                                                                              iteration,
                                                                              t,
                                                                              th_gap))
            iteration += 1

            with open(self.csv_name, 'a') as f:
                f.write("{},{},{},{},{},{},{}\n".format(t, best, mean, worst, variance, mean_ar, std_ar))

            if iteration == 200:
                break

        return


if __name__ == '__main__':

    env = gym.make("CartPole-v0")
    state_dim = 4
    feature_means = None

    gamma = 0.99
    epsilon = 0.1

    rb_dim = 5
    #rb_bfopt = "deep_cartpole"
    rb_bfopt = "gaussian_sum"

    lspi_bfdim = 5
    #lspi_bfopt = "deep_cartpole"
    #lspi_bfopt = "gaussian_sum"
    lspi_bfopt = "dan_pred"

    num_traj_for_policy = 100
    num_expert = 100
    num_traj_for_mu = 100

    reward_basis = RewardBasis(state_dim, rb_dim, gamma, feature_means, bfopt=rb_bfopt)

    iteration_names = ["DEBUG"]
    for debug_name in iteration_names:
        irl = IRL(env, reward_basis, gamma, epsilon, debug_name,
                        lspi_bfdim=lspi_bfdim,
                        lspi_bfopt=lspi_bfopt,
                        num_traj_for_policy=num_traj_for_policy,
                        num_expert=num_expert,
                        num_traj_for_mu=num_traj_for_mu,
                        num_eval=100)
        irl.loop()
