import numpy as np
import ipdb

class Basis_Function:
    """
    Basis function.
    """
    def __init__(self, input_dim, num_features, num_actions, gamma):
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma # 5?
        self.num_actions = num_actions
        self.num_feature_means = self.num_features - 1

        self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.num_feature_means)] 
        # (num_means, input_dim) # (state_dim, state_dim)

    def _num_basis(self):
        return self.num_features * self.num_actions


    def __calc_basis_component(self, state, mean, gamma):
        """
        Calculate basis component. exp(-gamma * sum ( diff^2 ))
        ~ exp(-gamma * sum(square error))

        less than 1, greater than 0
        1 >= np.exp(-gamma * np.sum(mean_diff)) > 0
        """
        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff))
    
    def evaluate(self, state, action):
        """
        state. 
        action.
        """
        if type(action) != int and type(action) != np.int64:
            raise ValueError("action should be int type!")
        if state.shape != self.feature_means[0].shape:
            print("state.shape : %d, self.means[0].shape : %d" % 
                    (state.shape, self.feature_means[0].shape))
            raise ValueError('Dimensions of state no match dimensions of means')

        k = self._num_basis()
        phi = np.zeros((k,))
        offset = self.num_features * action

        rbf = [self.__calc_basis_component(state, mean, self.gamma) for mean in self.feature_means]
        phi[offset] = 1.
        phi[offset + 1: offset + 1 + len(rbf)] = rbf

        return phi

    def evaluate_state(self, state):

        rbf = [self.__calc_basis_component(state, mean, self.gamma) for mean in self.feature_means]
        phi_state = [1.]
        phi_state += rbf

        return phi_state
