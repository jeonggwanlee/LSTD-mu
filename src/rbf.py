import numpy as np
import ipdb
from deep_cartpole import DeepCartPole

class Basis_Function:
    """
    Basis function.
    """
    def __init__(self, input_dim, num_features, num_actions, gamma, opt, center_opt="random"):
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.num_actions = num_actions
        self.opt = opt
        assert self.opt in ["gaussian", "gaussian_sum", "sigmoid", "sigmoid_sum", "mixture_gaussian_square", "square", "deep_cartpole"]
        assert center_opt in ["random", "fix"]

        self.feature_means2 = None
        self.dcp = None

        if self.opt == "mixture_gaussian_square":
            assert self.num_features == 13
            self.num_feature_means = 4
            self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.num_feature_means)]
            self.feature_means2 = [np.random.uniform(-1, 1, input_dim) for _ in range(self.num_feature_means)]

        elif self.opt == "square":
            assert self.num_features == 5
            self.num_feature_means = 4
            self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.num_feature_means)]

        elif self.opt == "deep_cartpole":
            #assert self.num_features == 4
            self.dcp = DeepCartPole()
            # fake
            #self.num_feature_means = 3
            self.dcp_output_dim = 3
            self.num_feature_means = self.num_features - 1
            self.feature_means = [np.random.uniform(-1, 1, self.dcp_output_dim) for _ in range(self.num_feature_means)]

        else:
            self.num_feature_means = self.num_features - 1  # for default value 1

            if center_opt == "random":
                self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.num_feature_means)]
            elif center_opt == "fix":
                assert self.num_feature_means in [1, 16]
                if self.num_feature_means == 1:
                    self.feature_means = [np.array([0, 0, 0, 0])]
                elif self.num_feature_means == 16:
                    fms = []
                    sort = [-1/3, 1/3]
                    for i0 in range(2):
                        for i1 in range(2):
                            for i2 in range(2):
                                for i3 in range(2):
                                    fm = np.array([sort[i0], sort[i1], sort[i2], sort[i3]])
                                    fms.append(fm)
                    self.feature_means = fms
     
        # (num_means, input_dim) # (state_dim, state_dim)

    def _num_basis(self):
        if self.opt == "gaussian_sum" or self.opt == "sigmoid_sum":
            return self.num_features * self.num_actions
        elif self.opt == "gaussian" or self.opt == "sigmoid":
            return self.num_features * self.input_dim * self.num_actions
        elif self.opt == "mixture_gaussian_square":
            return self.num_features * self.num_actions
        elif self.opt == "square":
            return self.num_features * self.num_actions
        elif self.opt == "deep_cartpole":
            return self.num_features * self.num_actions

    def __calc_basis_component_by_gaussian_sum(self, state, mean, gamma):
        """
        Calculate basis component. exp(-gamma * sum ( diff^2 ))
        ~ exp(-gamma * sum(square error))

        less than 1, greater than 0
        1 >= np.exp(-gamma * np.sum(mean_diff)) > 0
        """
        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff))

    def __calc_basis_component_by_gaussian(self, state, mean, gamma):
        mean_diff = (state - mean)**2
        return np.exp(-gamma * mean_diff)

    def __calc_basis_component_by_sigmoid(self, state, mean):
        """

        """
        mean_diff = abs(state - mean)
        value = np.array([self.logistic_sigmoid(a) for a in mean_diff])
        return value

    def __calc_basis_component_by_sigmoid_sum(self, state, mean):
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
            offset = self.num_features * self.input_dim * action

            rbf = [self.__calc_basis_component_by_gaussian(state, mean, self.gamma) for mean in self.feature_means]
            rbf2 = np.array(rbf)
            rbf3 = np.reshape(rbf2, [-1])
            phi[offset:offset+self.input_dim] = 1.
            phi[offset + self.input_dim: offset + self.input_dim + len(rbf3)] = rbf3

        elif self.opt == "gaussian_sum":

            k = self._num_basis()
            phi = np.zeros((k,))
            offset = self.num_features * action

            rbf = [self.__calc_basis_component_by_gaussian_sum(state, mean, self.gamma) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset + 1: offset + 1 + len(rbf)] = rbf

        elif self.opt == "sigmoid":

            k = self._num_basis()
            #num_state = len(state)
            #k = self.num_feature_means * num_state * self.num_actions
            phi = np.zeros((k,))
            offset = self.num_features * self.input_dim * action

            rbf = [self.__calc_basis_component_by_sigmoid(state, mean) for mean in self.feature_means]
            rbf2 = np.array(rbf)
            rbf3 = np.reshape(rbf2, [-1])
            phi[offset:offset+self.input_dim] = 1.
            phi[offset + self.input_dim: offset + self.input_dim + len(rbf3)] = rbf3

        elif self.opt == "sigmoid_sum":
            k = self._num_basis()
            phi = np.zeros((k,))
            offset = self.num_features * action

            rbf = [self.__calc_basis_component_by_sigmoid_sum(state, mean) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset + 1: offset + 1 + len(rbf)] = rbf

        elif self.opt == "mixture_gaussian_square":
            k = self._num_basis()
            phi = np.zeros((k,))
            offset = self.num_features * action
            rbf = [self.__calc_basis_component_by_gaussian_sum(state, mean, self.gamma) for mean in self.feature_means]
            rbf2 = [self.__calc_basis_component_by_sigmoid_sum(state, mean) for mean in self.feature_means2]
            phi[offset] = 1.
            phi[offset + 1: offset + 1 + len(rbf)] = rbf
            #phi[offset + 1 + len(rbf): offset + 1 + len(rbf) + 4] = state
            phi[offset + 1 + len(rbf) + 4: offset + 1 + len(rbf) + 4 + 4] = rbf2
            phi[offset + 1 + len(rbf): offset + 1 + len(rbf) + 4] = np.array([0,0,0,0])
            #phi[offset + 1 + len(rbf) + 4: offset + 1 + len(rbf) + 4 + 4] = np.array([0,0,0,0])
 
        elif self.opt == "square":
            k = self._num_basis()
            phi = np.zeros((k,))

            offset = self.num_features * action
            phi[offset] = 1.
            phi[offset + 1: offset + 1 + 4] = state**2

        elif self.opt == "deep_cartpole":
            k = self._num_basis()
            phi = np.zeros((k,))

            dcp_output = self.dcp.get_features(state)

            # ipdb.set_trace()
            offset = self.num_features * action
            rbf = [self.__calc_basis_component_by_gaussian_sum(dcp_output, mean, self.gamma) for mean in self.feature_means]
            phi[offset] = 1.
            phi[offset+1: offset + 1 + len(rbf)] = rbf

        return phi

    def evaluate_state(self, state):

        rbf = [self.__calc_basis_component(state, mean, self.gamma) for mean in self.feature_means]
        phi_state = [1.]
        phi_state += rbf

        return phi_state
