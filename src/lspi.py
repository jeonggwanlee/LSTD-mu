import numpy as np
from rbf import Basis_Function
from policy import Policy

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
    def __init__(self, num_actions=3, num_means=2, gamma=0.99):
        """
        num_actions. Number of actions. Int.
        num_means. Number of means. Int.
        gamma. Float.
        """
        
        print ("LSPI init!")
        print ("num_actions : %d, num_means(dim_of_states) : %d" %
                (num_actions, num_means))

        # 6, 6, 1, gamma
        """
        basis_function

        """
        self.basis_function = Basis_Function(num_means, num_means, num_actions, gamma)
        num_basis = self.basis_function._num_basis()

        self.policy = Policy(self.basis_function, num_basis)
        self.lstdq = LSTDQ(self.basis_function, gamma, self.policy)

        self.stop_criterion = 10**-5
        self.gamma = gamma

    def _act(self, state):
        index = self.policy.get_actions(state)
        action = self.policy.actions[index[0]]
        return action

    def train(self, sample, total_iteration, w_important_Sampling=False):
        
        error = float('inf')
        num_iteration = 0
        epsilon = 0.001
        
        while (epsilon * (1 - self.gamma) / self.gamma) < error and num_iteration < total_iteration:

            if w_important_Sampling:
                new_weights = self.lstdq.train_parameter(sample,
                                                         self.policy,
                                                         self.basis_function)
                #new_weights = self.lstdq.train_weight_parameter(sample,
                #                                                self.policy,
                #                                                self.basis_function)

            else:
                new_weights = self.lstdq.train_parameter(sample,
                                                                self.policy,
                                                                self.basis_function)

            # difference between current policy and target policy
            error = np.linalg.norm((new_weights - self.policy.weights))
            
            self.policy.theta_behavior = self.policy.weights
            self.policy.weights = new_weights
            num_iteration += 1

        return self.policy
            

class LSTDQ:
    def __init__(self, basis_function, gamma, init_policy):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = init_policy
        self.greedy = []

    def train_parameter(self, sample, policy, basis_function):
        """ Compute Q value function of current policy
            to obtain the greedy policy
            -> theta
        """
        k = basis_function._num_basis()

        A = np.zeros([k, k])
        b = np.zeros([k, 1])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2]
        next_states = sample[3]

        for i in range(len(states)):
            # take action from the greedy target policy
            index = policy.get_actions(next_states[i]) # TODO: validation in case more actions
            # index has base actions
            action = policy.actions[index[0]]
            
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

    #def train_weight_parameter(self, sample, policy, basis_function):
    #    """ Compute Q value function of current policy
    #        to obtain the greedy policy
    #    """

     #   k = basis_function._num_basis()
     #   A = np.zeros([k, k])
