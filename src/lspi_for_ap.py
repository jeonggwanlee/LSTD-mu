import numpy as np
from rbf import Basis_Function
from policy import Policy
import ipdb

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

class LSPI_AP:
    """ Learns a policy from samples
        D : source of samples (s, a, r, s`)
        k : number of basis functions
        phi : basis functions
        gamma : discount factor
        epsilon : stopping criterion
        policy(pi) : initial policy
    """
    def __init__(self, num_actions=3, state_dim=2, gamma=0.99):
        """
        num_actions. Number of actions. Int.
        state_dim. Number of means. Int. (= state_dim)
        gamma. Float.
        """
        
        print ("LSPI init!")
        print ("num_actions : %d, state_dim(dim_of_states) : %d" %
                (num_actions, state_dim))

        num_features = state_dim + 1 # for convenience 
        self.basis_function = Basis_Function(state_dim, num_features, num_actions, gamma)

        num_basis = self.basis_function._num_basis()
        actions = list(range(num_actions))
        self.policy = Policy(self.basis_function, num_basis, actions)
        self.lstdq = LSTD_mu(self.basis_function, gamma, self.policy)
        self.stop_criterion = 10**-5
        self.gamma = gamma

    def act(self, state):
        """ phi(s) = argmax_a Q(s, a) and pick first action
        """
        index = self.policy.get_actions(state)
        action = self.policy.actions[index[0]]
        return action

#    def train(self, sample, total_iteration, w_important_Sampling=False):
    def train(self, sample, w_important_Sampling=False):
        
        error = float('inf')
        num_iteration = 0
        epsilon = 0.001

        if w_important_Sampling:
            new_weights = self.lstdq.train_weight_parameter(sample, self.policy)
        else:
            new_weights = self.lstdq.train_parameter(sample, self.policy)

        error = np.linalg.norm((new_weights - self.policy.weights))
        #print("weight shift norm : ", error)
        self.policy.update_weights(new_weights)

        #while (epsilon * (1 - self.gamma) / self.gamma) < error and num_iteration < total_iteration:
        #    if w_important_Sampling:
        #        new_weights = self.lstdq.train_weight_parameter(sample, self.policy)
        #    else:
        #        new_weights = self.lstdq.train_parameter(sample, self.policy)

        #    error = np.linalg.norm((new_weights - self.policy.weights))
        #    print("weight shift norm : ", error)
        #    self.policy.update_weights(new_weights)
        #    num_iteration += 1

        return error


class LSTD_mu:
    def __init__(self, basis_function, gamma, init_policy):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = init_policy
        self.greedy = [] # for train weight parameter

    def train_parameter(self, sample, policy):
        """ Compute Q value function of current policy
            to obtain the greedy policy
            -> theta
        """
    
        self.policy = policy
        k = self.basis_function._num_basis()

        A = np.zeros([k, k])
        b = np.zeros([k, 1])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2]
        next_states = sample[3]

        SAMPLE_SIZE = len(states)
        for i in range(SAMPLE_SIZE):
            # take action from the greedy target policy
            action = self.policy.get_best_action(next_states[i])

            phi =      self.basis_function.evaluate(states[i], actions[i])
            phi_next = self.basis_function.evaluate(next_states[i], action)
            
            loss = (phi - self.gamma * phi_next)
            phi  = np.resize(phi, [k, 1])
            loss = np.resize(phi, [1, len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])
        #end for i in range(len(states)):

        inv_A = np.linalg.inv(A)
        theta = np.dot(inv_A, b)

        return theta

    def train_weight_parameter(self, sample, policy):
        """ Compute Q value function of current policy
            to obtain the greedy policy
        """

        k = self.basis_function._num_basis()
        A = np.zeros([k, k])
        b = np.zeros([k, 1])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2]
        next_states = sample[3]
        
        SAMPLE_SIZE = len(states)
        self.greedy = np.zeros_like(actions)
        self.greedy = np.reshape(self.greedy, [1, len(actions)])
        self.greedy = self.greedy[0] # (?, )

        sum_W = 0.0
        W = 1.0
        for i in range(SAMPLE_SIZE):
            pi_state_best = self.policy.get_best_action(states[i])                # pi(s)^{*} == argmax_{a} Q(s, a)
            prob_target = self.policy.q_value_function(states[i], pi_state_best)  # Q(s, pi(s)^{*})
            prob_behavior = self.policy.behavior(states[i], actions[i])           # \hat{Q}(s, a)

            if prob_behavior == 0.0:
                W = 0
            else:
                W = (prob_target / prob_behavior)
                sum_W = sum_W + W

        for i in range(SAMPLE_SIZE):
            pi_next_state_best = self.policy.get_best_action(next_states[i])  # max pi(s') == argmax_{a'} Q(s', a')
            phi = self.basis_function.evaluate(states[i], actions[i])                   # phi(s, a)
            phi_next = self.basis_function.evaluate(next_states[i], pi_next_state_best) # phi(s', pi(s')^{*})

            pi_state_best = self.policy.get_best_action(states[i])                 # pi(s)^{*}
            prob_target = self.policy.q_value_function(states[i], pi_state_best)   # Q(s, pi(s)^{*})
            prob_behavior = self.policy.behavior(states[i], actions[i])            # \hat{Q}(s, a)

            self.greedy[i] = pi_state_best

            exp = i - SAMPLE_SIZE   #[-SAMPLE_SIZE, ...]
            norm_W = (prob_target / prob_behavior) / sum_W   # (Q(s, pi(s)^{*}) / \hat{Q}(s, a)) / sum_W

            # important weighting on the whole transition
            loss = norm_W * (phi - self.gamma * phi_next)

            phi = np.resize(phi, [k, 1])
            loss = np.resize(phi, [1, len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])

        inv_A = np.linalg.inv(A)
        theta = np.dot(inv_A, b)

        return theta



