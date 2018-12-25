import pickle
import ipdb
import gym

from reward_basis import RewardBasis, Theta
from record import get_test_record_title


def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    v = np.zeros(n_states)

    diff = float("inf")
    # debug
    iter = 0
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v
        iter += 1
    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """

    -> Q. shape = (n_states, n_actions)
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])



class IRL:
    def __init__(self, reward_basis, theta, initial_policy, expert_trajectories, gamma, state0):
        self.reward_basis = reward_basis
        self.initial_policy = initial_policy
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma
        self.mu_expert = self.get_expert_feature_expectation(self.expert_trajectories)
        self.theta = theta
        self.state0 = state0

    def get_initial_fe(self):
        phi_s0 = irl.reward_basis.evaluate(self.state0)
        R = np.dot(phi, irl.theta.weights)

    def initial_policy(self, env):
         

    def get_expert_feature_expectation(self, expert_trajectories):
        """ expert_trajectories. (num_traj, traj_len, 5) = (100, 200, 5)
            
        """
        # [[state, action, reward, next_state, done] ...]

        mu_expert_sum = None
        for i, one_traj in enumerate(expert_trajectories):

            one_mu = None
            gamma_update = 1.0 / self.gamma
            for j, sample in enumerate(one_traj): # [s, a, r, s', d]
                state = sample[0]
                phi_state = self.reward_basis.evaluate(state)
                gamma_update *= self.gamma
                phi_time_unit = phi_state * gamma_update
                if j == 0:
                    one_mu = phi_time_unit
                else:
                    one_mu += phi_time_unit
            
            if i == 0:
                mu_expert_sum = one_mu
            else:
                mu_expert_sum += one_mu

        mu_expert = mu_expert_sum / len(expert_trajectories)
        return mu_expert



    #def get_feature_expectation(self, state,):


if __name__ == '__main__':
    exp_name = get_test_record_title("CartPole-v0", 1000, 'initial2', num_tests=1, important_sampling=True)
    traj_name = exp_name + '_num_traj100_pickle.bin'
    with open(traj_name, 'rb') as rf:
        trajs = pickle.load(rf) #[[state, action, reward, next_state, done], ...]

    #best_agent = get_best_agent('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True)
    state_dim = 4
    num_basis = 10
    feature_means_name = "reward_basis_statedim{}_numbasis{}_pickle.bin".format(state_dim,
                                                                                num_basis)
    with open(feature_means_name, 'rb') as rf:
        feature_means = pickle.load(rf)

    gamma = 0.99
    reward_basis = RewardBasis(state_dim, num_basis, gamma, feature_means)
    theta = Theta(num_basis)

    env = gym.make("CartPole-v0")
    state0 = env.reset()
    initial_policy = None
    irl = IRL(reward_basis, theta, initial_policy, trajs, gamma, state0) 
    ipdb.set_trace()
