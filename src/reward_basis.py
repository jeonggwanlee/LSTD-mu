import gym
import numpy as np
import pickle
import os
import ipdb

from test import get_best_agent
from record import get_test_record_title


class RewardBasis:
    def __init__(self, state_dim, num_basis, gamma=0.99, feature_means=None):
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.gamma = gamma

        if feature_means:
            if len(feature_means) == self.num_basis:
                if len(feature_means[0]) == self.state_dim:
                    self.feature_means = feature_means
                else:
                    raise ValueError("len(feature_means[0]){state_dim} != self.state_dim")
            else:
                raise ValueError("len(feature_means) != self.num_basis")
        else:
            self.feature_means = [np.random.uniform(-1, 1, self.state_dim) for _ in range(self.num_basis)]
            #save
            feature_means_name = "reward_basis_statedim{}_numbasis{}_pickle.bin".format(self.state_dim, self.num_basis)
            if os.path.exists(feature_means_name):
                print("{} already exists, could you remove?".format(feature_means_name))
                answer = input()
                if answer == 'y' or answer == 'Y':
                    os.remove(feature_means_name)

            with open(feature_means_name, 'wb') as wf:
                pickle.dump(self.feature_means, wf)

    def _num_basis(self):
        return self.num_basis

    def __calc_basis_component(self, state, mean, gamma):

        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff))

    def evaluate(self, state):

        p = self._num_basis()
        phi = np.zeros((p,))
        basis_function = [self.__calc_basis_component(state, mean, self.gamma) for mean in self.feature_means]
        phi[0:len(basis_function)] = basis_function

        return phi

    def evaluate_multi_states(self, states):

        phi_stack = None
        num_states = states.shape[0]
        for i in range(num_states):
            phi = self.evaluate(states[i])
            if i == 0:
                phi_stack = phi
            else:
                phi_stack = np.vstack((phi_stack, phi))

        return phi_stack
            

if __name__ == "__main__":
   
    exp_name = get_test_record_title("CartPole-v0", 1000, 'initial2', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_num_traj100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        trajs = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    #best_agent = get_best_agent('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True)
    state_dim = 4
    num_basis = 9
    feature_means_name = "CartPole-v0_RewardBasis{}_pickle.bin".format(num_basis)
    with open(feature_means_name, 'rb') as rf:
        feature_means = pickle.load(rf)
    reward_basis = RewardBasis(state_dim, num_basis, feature_means)
    ipdb.set_trace()
