import numpy as np
import ipdb

class Policy:
    def __init__(self, basis, num_theta, actions, theta=None):
        """
        self.basis_function
        self.weights
        """
        self.basis_function = basis
        self.actions = actions
        self.num_theta = num_theta
        self.theta_behavior = theta

        if theta is None:
            self.weights = np.random.uniform(-1.0, 1.0, size=(num_theta, 1))
        else:
            self.weights = theta

    def behavior(self, state, action):
        if self.theta_behavior is None:
            self.theta_behavior = np.random.uniform(-1.0, 1.0, size=(self.num_theta,))

        vector_basis = self.basis_function.evaluate(state, action)
        return np.dot(vector_basis, self.theta_behavior) # Phi * w

    def q_value_function(self, state, action):
        """ Q(s, a; w) = sum (pi(s, a) * weights)
        """
        vector_basis = self.basis_function.evaluate(state, action) # basis functions pi(s, a)
        return np.dot(vector_basis, self.weights)  # pi(s, a) * weights

    def update_weights(self, new_weights):
        self.theta_behavior = self.weights
        self.weights = new_weights

    # LSPI._act
    def get_actions(self, state):
        """ pi(s) = argmax_a Q(s, a)
        state.
        -> best_actions
        """
        q_state_action = [self.q_value_function(state, a) for a in self.actions]
        q_state_action = np.reshape(q_state_action, [len(q_state_action), 1]) # convert to column vector

        index = np.argmax(q_state_action)
        q_max = q_state_action[index]
        best_actions = [self.actions[index]]
        ind = [index]

        # Find other actions which has same value of q_max
        for i in range(len(q_state_action)):
            if q_state_action[i] == q_max and index != i:
                best_actions.append(self.actions[i])
                ind.append(i)

        return best_actions

    def get_best_action(self, state):
        """ pi(s) = argmax_a Q(s, a)
        
        """
        indices = self.get_actions(state)
        return self.actions[indices[0]]
