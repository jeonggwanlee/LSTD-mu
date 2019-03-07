import numpy as np
import ipdb

from deep_cartpole import DeepCartPole
from deep_action_network import DeepActionNetwork

class Basis_Function:
    """
    Basis function.
    """
    def __init__(self, input_dim, num_features, num_actions, gamma, opt):
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.num_actions = num_actions
        self.opt = opt
        assert self.opt in ["gaussian", "sigmoid", "deep_cartpole", "dan_pred", "dan_h1", "dan_h2"]
        self.dcp = None
        self.dan = None

        if self.opt == "deep_cartpole":
            self.dcp = DeepCartPole()
            self.dcp_output_dim = 3
            self.num_feature_means = self.num_features - 1
            self.feature_means = [np.random.uniform(-1, 1, self.dcp_output_dim) for _ in range(self.num_feature_means)]
        elif self.opt[:3] == "dan":
            feature_op = self.opt[4:]
            self.dan = DeepActionNetwork(feature_op, 4, 2, 10, 10)
            if not self.dan.isRestore():
                self.dan.learn()
            feature_dim = self.dan.get_feature_dim()
            self.num_feature_means = self.num_features - 1
            self.feature_means = [np.random.uniform(-1, 1, feature_dim) for _ in range(self.num_feature_means)]
        else:
            self.num_feature_means = self.num_features - 1  # for default value 1
            self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.num_feature_means)]
        
        # (num_means, input_dim) # (state_dim, state_dim)

    def _num_basis(self):
        return self.num_features * self.num_actions

    def __calc_basis_component_by_gaussian(self, state, mean, gamma):
        """
        Calculate basis component. exp(-gamma * sum ( diff^2 ))
        ~ exp(-gamma * sum(square error))

        less than 1, greater than 0
        1 >= np.exp(-gamma * np.sum(mean_diff)) > 0
        """
        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff))

    def __calc_basis_component_by_sigmoid(self, state, mean):
        norm = np.linalg.norm(state - mean)
        return self.logistic_sigmoid(norm)

    def logistic_sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def evaluate(self, state, action):
        """
        state.
        action.
        """
        if type(action) != int and type(action) != np.int64:
            raise ValueError("action should be int type!")
        #if state.shape != self.feature_means[0].shape:
        #    print("state.shape : %d, self.means[0].shape : %d" %
        #            (state.shape, self.feature_means[0].shape))
        #    raise ValueError('Dimensions of state no match dimensions of means')

        if self.opt == "gaussian":

            k = self._num_basis()
            phi = np.zeros((k,))
            offset = self.num_features * action

            rbf = [self.__calc_basis_component_by_gaussian(state, mean, self.gamma) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset + 1: offset + 1 + len(rbf)] = rbf

        elif self.opt == "sigmoid":
            k = self._num_basis()
            phi = np.zeros((k,))
            offset = self.num_features * action

            rbf = [self.__calc_basis_component_by_sigmoid(state, mean) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset + 1: offset + 1 + len(rbf)] = rbf

        elif self.opt == "deep_cartpole":
            k = self._num_basis()
            phi = np.zeros((k,))

            dcp_output = self.dcp.get_features(state)

            # ipdb.set_trace()
            offset = self.num_features * action
            rbf = [self.__calc_basis_component_by_gaussian(dcp_output, mean, self.gamma) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset+1: offset + 1 + len(rbf)] = rbf

        elif self.opt[:3] == "dan":
            k = self._num_basis()
            phi = np.zeros((k,))

            #q_value = self.dan.get_action_pred(state)
            q_value = self.dan.get_features(state)

            offset = self.num_features * action
            rbf = [self.__calc_basis_component_by_gaussian(q_value, mean, self.gamma) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset+1: offset+1+len(rbf)] = rbf

        return phi

    def evaluate_state(self, state):

        rbf = [self.__calc_basis_component(state, mean, self.gamma) for mean in self.feature_means]
        phi_state = [1.]
        phi_state += rbf

        return phi_state
