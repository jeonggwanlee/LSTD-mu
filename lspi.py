import numpy as np
import pickle
import os
import ipdb

from policy import Policy
from rbf import Basis_Function

"""
important property of LSPI is that it does not require an an approximate
policy representation, At each iteration, a different policy is evaluated
and certain sets of basis functions may be more appropriate than others 
for representing the state-action value function for each of these
policies.

since LSPI approximates state-action value functions,
it can use samples from any policy to estimate the state-action value
function of another policy.
This focuses attention more clearly on the issue of exploration 
since any policy can be followed while collecting samples.
"""

class LSPI:
    """ Learns a policy from samples
        D : source of samples (s, a, r, s`)
        k : number of basis functions
        phi : basis functions
        gamma : discount factor
        epsilon : stopping criterion
        policy(pi) : initial policy
    """
    def __init__(self, num_actions=2, state_dim=4, basis_function_dim=5, gamma=0.99, opt="sigmoid"):
        """
        num_actions. Number of actions. Int.
        state_dim. Number of means. Int. (= state_dim)
        gamma. Float.
        """
        # num_features = state_dim + 1  # for convenience
        basis_function_pickle_name = "LSPI_bf_#State{}_#Features{}_#Action{}_Opt{}.pickle".format(
                                                                                        state_dim,
                                                                               basis_function_dim,
                                                                                      num_actions,
                                                                                              opt)

        print("basis_function_pickle_name : {}".format(basis_function_pickle_name))
        self.basis_function = Basis_Function(state_dim, basis_function_dim, num_actions, gamma, opt)

        self.num_basis = self.basis_function._num_basis()
        self.actions = list(range(num_actions))
        self.policy = Policy(self.basis_function, self.num_basis, self.actions)
        self.lstdq = LSTDQ(self.basis_function, gamma)
        self.stop_criterion = 10**-5
        self.gamma = gamma

    def initialize_policy(self):
        self.policy = Policy(self.basis_function, self.num_basis, self.actions)

    def act(self, state):
        """ phi(s) = argmax_a Q(s, a) and pick first action
        """
        index = self.policy.get_actions(state)
        action = self.policy.actions[index[0]]
        return action

    def train(self, sample, w_important_sampling=False):

        error = float('inf')
        iter = 0
        lspi_iteration = 20
        epsilon = 0.00001

        policy_list = []
        error_list = []

        while epsilon < error and iter < lspi_iteration:
            policy_list.append(self.policy.weights)
            if w_important_sampling:
                new_weights = self.lstdq.train_weight_parameter(sample, self.policy)
            else:
                new_weights = self.lstdq.train_parameter(sample, self.policy)

            error = np.linalg.norm((new_weights - self.policy.weights))
            error_list.append(error)
            #print("train error {} : {}".format(iter, error))
            self.policy.update_weights(new_weights)
            iter += 1

        return error_list


class LSTDQ:
    def __init__(self, basis_function, gamma):
        self.basis_function = basis_function
        self.gamma = gamma

    def train_parameter(self, sample, greedy_policy):

        """ Compute Q value function of current policy
            to obtain the greedy policy
            -> theta
        """
        p = self.basis_function._num_basis()

        A = np.zeros([p, p])
        b = np.zeros([p, 1])
        np.fill_diagonal(A, .1)  # Singular matrix error

        states = sample[0]
        actions = sample[1]
        rewards = sample[2]
        next_states = sample[3]

        SAMPLE_SIZE = len(states)
        for i in range(SAMPLE_SIZE):
            phi = self.basis_function.evaluate(states[i], actions[i])

            greedy_action = greedy_policy.get_best_action(next_states[i])
            phi_next = self.basis_function.evaluate(next_states[i], greedy_action)

            loss = (phi - self.gamma * phi_next)
            phi = np.reshape(phi, [p, 1])
            loss = np.reshape(loss, [1, p])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])
        #end for i in range(len(states)):
 
        inv_A = np.linalg.inv(A)
        w = np.dot(inv_A, b)

        return w

    def train_weight_parameter(self, sample, greedy_policy):
        """ Compute Q value function of current policy
            to obtain the greedy policy
        """

        p = self.basis_function._num_basis()
        A = np.zeros([p, p])
        b = np.zeros([p, 1])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2]
        next_states = sample[3]
        
        SAMPLE_SIZE = len(states)

        sum_W = 0.0
        W = 1.0
        for i in range(SAMPLE_SIZE):
            greedy_action = greedy_policy.get_best_action(states[i])                # pi(s)^{*} == argmax_{a} Q(s, a)
            prob_target = greedy_policy.q_value_function(states[i], greedy_action)  # Q(s, pi(s)^{*})
            prob_behavior = greedy_policy.behavior(states[i], actions[i])           # \hat{Q}(s, a)

            if prob_behavior == 0.0:
                W = 0
            else:
                W = (prob_target / prob_behavior)
                sum_W = sum_W + W

        for i in range(SAMPLE_SIZE):
            greedy_next_action = greedy_policy.get_best_action(next_states[i])
            phi = self.basis_function.evaluate(states[i], actions[i])
            phi_next = self.basis_function.evaluate(next_states[i], greedy_next_action)

            greedy_action = greedy_policy.get_best_action(states[i])                 # pi(s)^{*}
            prob_target = greedy_policy.q_value_function(states[i], greedy_action)   # Q(s, pi(s)^{*})
            prob_behavior = greedy_policy.behavior(states[i], actions[i])            # \hat{Q}(s, a)

            norm_W = (prob_target / prob_behavior) / sum_W   # (Q(s, pi(s)^{*}) / \hat{Q}(s, a)) / sum_W

            # important weighting on the whole transition
            loss = norm_W * (phi - self.gamma * phi_next)

            phi = np.resize(phi, [p, 1])
            loss = np.resize(loss, [1, len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])

        inv_A = np.linalg.inv(A)
        theta = np.dot(inv_A, b)

        return theta



