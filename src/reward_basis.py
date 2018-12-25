
import numpy as np

class Theta:
    def __init__(self):


class RewardBasis:
    def __init__(self, num_basis):
        self.num_basis = num_basis
        self.num_basis_means = self.num_basis

        self.feature_means = [np.random.uniform(-1, 1, self.num_basis) for _ in range(self.num_basis)]

    def _num_basis(self):
        return self.num_basis

    def __calc_basis_compoment(self, state, gamma):

        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff))

    def evaluate(self, state):

        p = self._num_basis()
        phi = np.zeros((p,))
        basis_function = [self.__calc_basis_component(state, mean, self.gamma) for mean in self.feature_means]
        phi[0:len(basis_function)] = basis_function

        return phi

