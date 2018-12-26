import pickle
import ipdb
import gym
import numpy as np

from reward_basis import RewardBasis, Theta
from record import get_test_record_title
from replay_memory import Memory
from lspi_for_ap import LSPI_AP

TRANSITION = 15000
EPISODE = 1000
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000

class IRL:
    def __init__(self, reward_basis, theta, initial_policy, trajectories, gamma, state0, env):
        self.reward_basis = reward_basis
        self.initial_policy = initial_policy
        self.trajectories = expert_trajectories
        self.gamma = gamma
        self.theta = theta
        self.state0 = state0
        self.env = env
        self.epsilon = 0.1
        self.mu_expert = self.compute_feature_expectation(self.trajectories)
        self.initial_trajectories = self._generate_trajectories_from_initial_policy()
        self.mu_initial = self.compute_feature_expectation(self.initial_trajectories)
        self.mu_bar = self.mu_initial
        def experiment_setting(self):
            self.num_actions = self.env.action_space.n
            self.state_dim = self.env.observation_space.shape[0]
            action_dim = 1
            self.memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)
        self.experiment_setting()
        self.irl_main()

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
        """ trajectories. (num_traj, traj_len, 5) = (100, 200, 5)
            
         [[state, action, reward, next_state, done] ...]
        """

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
            
            if i == 0:
                mu_sum = one_mu
            else:
                mu_sum += one_mu
            # for j
        # for i
        mu = mu_sum / len(trajectories)
        return mu


    def irl_main(self):
        ##self.mu_initial (mu_0)
        # get t^1, mu^1

        ipdb.set_trace() 

        # compute_mu_bar
        """ while loop for policy iteration
            Ng & Abeel's paper., the projection margin method
        """
        theta_as_reward_space_param = self.mu_expert - self.mu_bar # theta
        t = np.linalg.norm(theta_as_reward_space_param, 2)
        print("Initial threshold: ", t)
        iteration = 0
        # R = pi w
        while t > self.epsilon:

            agent = LSPI_AP(self.num_actions, self.state_dim)
            best_agent = self.get_best_agent(self.env, self.memory, agent, 
                    theta_as_reward_space_param,
                    isRender=False)
            
            new_trajectories = self._generate_new_trajectories(best_agent, n_trajectories=1000)
            feature_expectations = self.compute_feature_expectation(new_trajectories)            

            # LSPI
            # w -> R -> Q -> mu -> updated_loss -> mu_bar -> w -> t
             



    def get_best_agent(self, env, memory, agent, theta, isRender=False):

    #state = sample[0]
    #phi_state = self.reward_basis.evaluate(state)
    #gamma_update *= self.gamma
    #phi_time_unit = phi_state * gamma_update
        Best_agent = None
        Best_mean_reward = -4000
        mean_reward = -4000
        
        for i in range(EPISODE):
            env.seed()
            state = env.reset()

            for j in range(TRANSITION):
                action = env.action_space.sample()
                next_state, _, done, info = env.step(action)
                phi_state = self.reward_basis.evaluate(state)
                memory.add([state, action, phi_state, next_state, done])
                state = next_state
                if done:
                    break

            if memory.container_size < BATCH_SIZE:
                sample = memory.select_sample(memory.container_size)
            else:
                sample = memory.select_sample(BATCH_SIZE)
            
            error = agent.train(sample, 20, True)
            
            reward_list = []
            for j in range(NUM_TESTS_MIDDLE):
                total_reward, _ = test_policy(env, agent)
                reward_list.append(total_reward)
            mean_reward = sum(reward_list) / NUM_TESTS_MIDDLE

            if Best_mean_reward < mean_reward:

                Best_agent = agent
                Best_mean_reward = mean_reward
                memory.clear_memory()
        # for i
                
        # Clean up
        memory.clear_memory()

        return Best_agent

if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 1000, 'initial2', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_num_traj100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        trajs = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    #best_agent = get_best_agent('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True)
    state_dim = 4
    num_basis = 10
    feature_means_name = "reward_basis_statedim{}_numbasis{}_pickle.bin".format(state_dim,
                                                                                num_basis)
    with open(feature_means_name, 'rb') as rf:
        feature_means = pickle.load(rf)

    gamma = 0.99
    reward_basis = RewardBasis(state_dim, num_basis, gamma, feature_means)
    theta = Theta(num_basis)

    env = gym.make("CartPole-v0")
    state0 = env.reset()
    initial_policy = None
    irl = IRL(reward_basis, theta, initial_policy, trajs, gamma, state0, env) 
    ipdb.set_trace()
