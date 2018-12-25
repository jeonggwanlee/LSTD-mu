

class IRL:
    def __init__(self, basis_function, initial_policy, expert_trajectories, gamma):
        self.basis_function = basis_function
        self.initial_policy = initial_policy
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma

    def get_feature_expectation(self, state,):
