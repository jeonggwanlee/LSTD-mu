import gym
import numpy as np
import os
import ipdb

from replay_memory import Memory
from lspi import LSPI
from record import get_test_record_title_v2
from test import test_policy

TRANSITION = 15000
MEMORY_SIZE = 50000
NUM_TRIALS = 100
NUM_TESTS_PER_TRIALS = 300

important_sampling = False
gamma = 0.99

def cartpole_experiment(basis_opt="gaussian_sum", basis_function_dim=5, saved_basis_use=True):
    print('Hello CartPole world!')
    env = gym.make('CartPole-v0')

    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    print("state_dim : %d\naction_dim : %d\nnum_actions : %d" % (state_dim, action_dim, num_actions))
    print("basis_opt : {}, basis_function_dim : {}".format(basis_opt, basis_function_dim))

    memory = Memory(MEMORY_SIZE, action_dim, state_dim)

    agent = LSPI(num_actions, state_dim, basis_function_dim, gamma=0.99, opt=basis_opt, saved_basis_use=saved_basis_use)
    return agent, env, memory, 'CartPole-v0'

###########################################################################################
def _random_sample(env, memory, agent, csv_error_name, episode, isRender=False):

    transition_iter_list = []
    for i in range(episode):
        state = env.reset()
        for j in range(TRANSITION):
            if isRender:
                env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                transition_iter_list.append(j)
                break
    middle_ti = sorted(transition_iter_list)[len(transition_iter_list)//2]
    average_ti = sum(transition_iter_list) // episode
    sum_ti = sum(transition_iter_list)

    sample = memory.select_sample(memory.container_size)
    error_list = agent.train(sample, important_sampling)
    error_list = list(map(str, error_list))
    error_str = ",".join(error_list)
    with open(csv_error_name, 'a') as f:
        f.write("{}\n".format(error_str))


    # Get middle mean
    rewards = []
    for j in range(NUM_TESTS_PER_TRIALS):
        total_reward = test_policy(env, agent, isRender=False)
        rewards.append(total_reward)
    mean = sum(rewards) / NUM_TESTS_PER_TRIALS
    best = max(rewards)
    worst = min(rewards)
    variance = np.var(rewards)
    print("mean reward : {}".format(mean))

    # Clean up
    memory.clear_memory()
    return mean, best, worst, variance, middle_ti, average_ti, sum_ti


def do_lspi(env, memory, agent, game_name, csv_error_name, episode):

    # Training
    middle, best, worst, variance, middle_ti, average_ti, sum_ti = _random_sample(env, memory, agent, csv_error_name, episode, isRender=False)
    std = pow(variance, 1/2)

    return middle, best, worst, std, sum_ti


def _main(episode_list, game_title='CartPole', basis_opt="gaussian_sum", basis_function_dim=5, saved_basis_use=True):
    if basis_function_dim < 2:
        print("basis_function_dim should be larger than 1.")
        exit()

    agent, env, memory, game_name = cartpole_experiment(basis_opt=basis_opt,
                                                        basis_function_dim=basis_function_dim,
                                                        saved_basis_use=saved_basis_use)

    experiment_title = get_test_record_title_v2(game_name,
                                                num_tests=NUM_TRIALS,
                                                important_sampling=important_sampling)
    print("Experiement title : {}".format(experiment_title))
    episode_list_str = list(map(str, episode_list))
    episodes_str = ','.join(episode_list_str)
    csv_name = experiment_title + 'BasisOpt{}_BasisFunctionDim{}_Episode{}_1010.csv'.format(basis_opt,
                                                                             basis_function_dim,
                                                                             episodes_str,
                                                                             )
    # third : fixed basis function
    csv_error_name = csv_name[:-4] + '_error.csv'
    print("csv_name : {}".format(csv_name))

    print('csv_file duplication check!')
    if os.path.isfile(csv_name):
        print('Do you want to remove csv_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(csv_name)
        else:
            raise ValueError('you should manually control it!')

    for episode in episode_list:
        print("episode {}".format(episode))
        middle_list = []
        best_list = []
        worst_list = []
        std_list = []
        sum_ti_list = []
        for i in range(NUM_TRIALS):
            if i % 10 == 0 and i != 0:
                print("trials ", i, "/", NUM_TRIALS)
                print("mid_avg : {} best_avg : {} worst_avg : {}".format(
                                                                sum(middle_list) / i,
                                                                sum(best_list) / i,
                                                                sum(worst_list) / i))
            #print("episode {}\n".format(episode))
            agent.initialize_policy()
            memory.clear_memory()
            middle, best, worst, std, sum_ti = do_lspi(env, memory, agent, game_name, csv_error_name, episode)
            middle_list.append(middle)
            best_list.append(best)
            worst_list.append(worst)
            std_list.append(std)
            sum_ti_list.append(sum_ti)

        middle_avg = sum(middle_list) / NUM_TRIALS
        best_avg = sum(best_list) / NUM_TRIALS
        worst_avg = sum(worst_list) / NUM_TRIALS
        std_avg = sum(std_list) / NUM_TRIALS
        sum_ti_avg = sum(sum_ti_list) / NUM_TRIALS

        with open(csv_name, 'a') as f:
            f.write("{},{},{},{},{}\n".format(best_avg, middle_avg, worst_avg, std_avg, sum_ti_avg))


if __name__ == '__main__':

    episode_list = [100]
    basis_options = ['gaussian', 'dan_h1', 'dan_pred']
    #_main(episode_list, game_title="CartPole", basis_opt="gaussian_sum", basis_function_dim=20, saved_basis_use=True, center_opt="random")
    #_main(episode_list, game_title="CartPole", basis_opt="mixture_gaussian_square", basis_function_dim=13, saved_basis_use=False, center_opt="random")
    #_main(episode_list, game_title="CartPole", basis_opt="square", basis_function_dim=5, saved_basis_use=False, center_opt="random")
    #_main(episode_list, game_title="CartPole", basis_opt="deep_cartpole", basis_function_dim=30, saved_basis_use=False, center_opt="random")
    #_main(episode_list, game_title="CartPole", basis_opt="dan_h1", basis_function_dim=30, saved_basis_use=False, center_opt="random")
    _main(episode_list, game_title="CartPole", basis_opt="dan_pred", basis_function_dim=10, saved_basis_use=False)
