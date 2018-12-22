import numpy as np

class Basis_Function:
    """
    Basis function.
    """
    def __init__(self, input_dim, num_means, num_actions, gamma):
        self.input_dim = input_dim
        self.num_means = num_means
        self.gamma = gamma # 5?
        self.num_actions = num_actions

        # np.random.uniform(low, high, size)
        self.means = [np.random.uniform(-1, 1, input_dim) for i in range(self.num_means)]

        self.beta = 8 # ?

    def _num_basis(self):
        """
        => (#states * #actions)
        """
        return (len(self.means)+1) * self.num_actions


    def __calc_basis_component(self, state, mean, gamma):
        """
        Calculate basis component. exp(-gamma * sum ( diff^2 ))
        ~ exp(-gamma * sum(square error))
        """
        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff))
    
    def evaluate(self, state, action):
        """
        state. 
        action.
        """
        if type(action) != int:
            action = action[0]

        if state.shape != self.means[0].shape:
            print("state.shape : %d, self.means[0].shape : %d" % 
                    (state.shape, self.means[0].shape))
            raise ValueError('Dimensions of state no match dimensions of means')

        phi = np.zeros((self._num_basis(),))
        offset = (len(self.means[0])+1) * action

        rbf = [self.__calc_basis_component(state, mean, self.gamma)
                for mean in self.means]
        try:
            phi[offset] = 1.
        except:
            import ipdb; ipdb.set_trace()
        phi[offset + 1: offset + 1 + len(rbf)] = rbf

        return phi
