""" Jeonggwan Lee (leejk526@kaist.ac.kr)
"""

import numpy as np
from policy import Policy
import ipdb

class LSTD_MU:
    def __init__(self, psi_function, gamma):
        self.psi_function = psi_function
        self.gamma = gamma

    def train_parameter_xi(self, sample, optimal_policy, reward_basis):
        """ Compute Q value function of current policy
            to obtain the greedy policy
            -> theta
        """
    
        q = self.psi_function._num_basis()
        p = reward_basis._num_basis()

        A = np.zeros([q, q])
        b = np.zeros([q, p])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        #rewards     = sample[2]
        next_states = sample[3]
        phi_stack   = reward_basis.evaluate_multi_states(states)

        SAMPLE_SIZE = len(states)
        for i in range(SAMPLE_SIZE):
            # take action from the greedy target policy
            #action = self.policy.get_best_action(next_states[i])
            action = optimal_policy.get_best_action(next_states[i])

            psi =      self.psi_function.evaluate(states[i], actions[i])
            psi_next = self.psi_function.evaluate(next_states[i], action)
            
            loss = (psi - self.gamma * psi_next)

            psi = np.resize(psi, [q, 1])
            loss = np.resize(loss, [1, q])
            phi = np.resize(phi_stack[i], [1, p])

            A = A + np.dot(psi, loss)
            b = b + (psi * phi)
        #end for i in range(len(states)):

        inv_A = np.linalg.inv(A)
        xi = np.dot(inv_A, b)
        assert xi.shape == (q, p)

        return xi

    def train_parameter_xi_with_important_sampling(self, sample, optimal_policy, reward_basis):
        """ Compute Q value function of current policy compared to prev policy
        """

        #self.policy = policy
        ##optimal_policy = policy
        #p = self.psi_function._num_basis()
        #q = reward_basis._num_basis()
        q = self.psi_function._num_basis()
        p = reward_basis._num_basis()

        #A = np.zeros([p, p])
        #b = np.zeros([p, q])
        A = np.zeros([q, q])
        b = np.zeros([q, p])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        #rewards     = sample[2]
        next_states = sample[3]
        phi_stack   = reward_basis.evaluate_multi_states(states)

        SAMPLE_SIZE = len(states)

        sum_W = 0.0
        W = 1.0
        for i in range(SAMPLE_SIZE):
            pi_state_best = optimal_policy.get_best_action(states[i])                # pi(s)^{*} == argmax_{a} Q(s, a)
            prob_target = optimal_policy.q_value_function(states[i], pi_state_best)  # Q(s, pi(s)^{*})
            prob_behavior = optimal_policy.behavior(states[i], actions[i])           # \hat{Q}(s, a)

            if prob_behavior == 0.0:
                W = 0
            else:
                W = (prob_target / prob_behavior)
                sum_W = sum_W + W

        for i in range(SAMPLE_SIZE):
            #pi_next_state_best = self.policy.get_best_action(next_states[i])  # max pi(s') == argmax_{a'} Q(s', a')
            pi_next_state_best = optimal_policy.get_best_action(next_states[i])  # max pi(s') == argmax_{a'} Q(s', a')
            psi = self.psi_function.evaluate(states[i], actions[i])                   # phi(s, a)
            psi_next = self.psi_function.evaluate(next_states[i], pi_next_state_best) # phi(s', pi(s')^{*})

            #pi_state_best = self.policy.get_best_action(states[i])                 # pi(s)^{*}
            #prob_target = self.policy.q_value_function(states[i], pi_state_best)   # Q(s, pi(s)^{*})
            #prob_behavior = self.policy.behavior(states[i], actions[i])            # \hat{Q}(s, a)

            pi_state_best = optimal_policy.get_best_action(states[i])                 # pi(s)^{*}
            prob_target = optimal_policy.q_value_function(states[i], pi_state_best)   # Q(s, pi(s)^{*})
            prob_behavior = optimal_policy.behavior(states[i], actions[i])            # \hat{Q}(s, a)

            # self.greedy[i] = pi_state_best

            # exp = i - SAMPLE_SIZE   #[-SAMPLE_SIZE, ...]
            norm_W = (prob_target / prob_behavior) / sum_W   # (Q(s, pi(s)^{*}) / \hat{Q}(s, a)) / sum_W

            # important weighting on the whole transition
            loss = norm_W * (psi - self.gamma * psi_next)

            #psi = np.resize(psi, [p, 1])
            psi = np.resize(psi, [q, 1])
            loss = np.resize(loss, [1, len(loss)])
            #phi = np.resize(phi_stack[i], [1, q])
            phi = np.resize(phi_stack[i], [1, p])

            A = A + np.dot(psi, loss)
            b = b + (psi * phi)

        inv_A = np.linalg.inv(A)
        xi = np.dot(inv_A, b)
        
        #assert xi.shape == (p, q)
        assert xi.shape == (q, p)
        return xi



