import pickle
import ipdb
import gym
import numpy as np
import copy

from reward_basis import RewardBasis, Theta
from record import get_test_record_title
from replay_memory import Memory
from lspi_for_ap import LSPI_AP

TRANSITION = 15000
EPISODE = 1000
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000
NUM_EVALUATION = 1000

class IRL:
    def __init__(self, env, reward_basis, expert_trajectories, gamma, epsilon):
        self.env = env
        self.reward_basis = reward_basis
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = None
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

        self.mu_expert = self.compute_feature_expectation(expert_trajectories)
        initial_trajectories = _generate_trajectories_from_initial_policy(self)
        self.mu_initial = self.compute_feature_expectation(initial_trajectories)
        self.mu_bar = self.mu_initial
        def __experiment_setting(self):
            self.num_actions = self.env.action_space.n
            self.state_dim = self.env.observation_space.shape[0]
            action_dim = 1
            self.memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)
        __experiment_setting(self)


    def _generate_new_trajectories(self, agent, n_trajectories=1000):
        trajectories = []
        for _ in range(n_trajectories):
            self.env.seed()
            state = self.env.reset()

            trajectory = []
            for _ in range(TRANSITION):
                action = agent._act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add([state, action, reward, next_state, done])
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
                if j == 0: one_mu = phi_time_unit
                else: one_mu += phi_time_unit
            # for j
            if i == 0: mu_sum = one_mu
            else: mu_sum += one_mu
        # for i
        mu = mu_sum / len(trajectories)
        return mu

    def _test_policy_with_approxi_reward(self, agent, reward_basis, theta, isRender=False):
        total_reward = 0.0
        self.env.seed()
        state = self.env.reset()
        for i in range(TRANSITION):
            if isRender:
                self.env.render()
            phi = reward_basis.evaluate(state)
            approxi_reward = np.dot(phi, theta)
            ipdb.set_trace()
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            total_reward += approxi_reward 
            if done:
                break

        return total_reward


    def _get_best_agent(self, memory, agent, theta, isRender=False):
        Best_agent = None
        Best_mean_reward = -4000
        mean_reward = -4000
        for _ in range(EPISODE):
            self.env.seed()
            state = self.env.reset()
            for j in range(TRANSITION):
                if isRender:
                    self.env.render()
                action = self.env.action_space.sample()
                next_state, _, done, _ = self.env.step(action)
                phi_state = self.reward_basis.evaluate(state)
                memory.add([state, action, phi_state, next_state, done])
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
                total_reward, = self._test_policy_with_approxi_reward(agent, 
                                                                      self.reward_basis,
                                                                      theta)
                reward_list.append(total_reward)
            mean_reward = sum(reward_list) / NUM_EVALUATION

            if Best_mean_reward < mean_reward:
                Best_agent = copy.deepcopy(agent)
                Best_mean_reward = mean_reward
                memory.clear_memory()
        # for i
        # Clean up
        memory.clear_memory()

        return Best_agent


    def loop(self):

        self.theta = self.mu_expert - self.mu_bar # theta
        t = np.linalg.norm(self.theta, 2)
        print("Initial threshold: ", t)
        iteration = 0
        ipdb.set_trace()
        # R = pi w
        while t > self.epsilon:

            agent = LSPI_AP(self.num_actions, self.state_dim)
            best_agent = self._get_best_agent(self.memory, agent, 
                                              self.theta, isRender=False)
            
            new_trajectories = self._generate_new_trajectories(best_agent, n_trajectories=1000)
            mu = self.compute_feature_expectation(new_trajectories)            
            updated_loss = mu - self.mu_bar
            self.mu_bar += updated_loss * updated_loss.dot(w) / np.square(updated_loss).sum()
            theta = self.mu_expert - self.mu_bar
            t = np.linalg.norm(theta, 2)
            iteration += 1
             
        return


if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 1000, 'initial2', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_num_traj100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        expert_trajectories = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    #best_agent = get_best_agent('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True)
    state_dim = 4
    num_basis = 10
    feature_means_name = "reward_basis_statedim{}_numbasis{}_pickle.bin".format(state_dim,
                                                                                num_basis)
    with open(feature_means_name, 'rb') as rf:
        feature_means = pickle.load(rf)

    env = gym.make("CartPole-v0")
    gamma = 0.99
    reward_basis = RewardBasis(state_dim, num_basis, gamma, feature_means)
    epsilon = 0.1

    irl = IRL(env, reward_basis, expert_trajectories, gamma, epsilon) 
    irl.loop()
    ipdb.set_trace()
