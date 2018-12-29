import gym
import numpy as np
import datetime
import ipdb
import os
import pickle
import copy

from replay_memory import Memory
from lspi import LSPI, LSTDQ
from record import get_test_record_title

TRANSITION = 15000
EPISODE = 1000
BATCH_SIZE = 300
MEMORY_SIZE = TRANSITION + 1000
NUM_EXPERI = 1
NUM_TESTS_MIDDLE = 200
TRAINOPT = ['random', 'initial', 'initial2', 'reuse']

important_sampling = True
lspi_iteration = 20 #?
gamma = 0.99

def experiment_1():
    print('Hello Acrobot world!')
    env = gym.make('Acrobot-v1')
    state = env.reset()

    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    print("state_dim : %d\naction_dim : %d\nnum_actions : %d" % (state_dim, action_dim, num_actions))

    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)

    agent = LSPI(num_actions, state_dim)
    return agent, env, memory, 'Acrobot-v1'

def experiment_2():
    print('Hello CartPole world!')
    env = gym.make('CartPole-v0')
    state = env.reset()

    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    print("state_dim : %d\naction_dim : %d\nnum_actions : %d" % (state_dim, action_dim, num_actions))

    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)

    agent = LSPI(num_actions, state_dim)
    return agent, env, memory, 'CartPole-v0'

def experiment_3():
    exit()
    # under construction
    print('Hello Pendulum world!')
    env = gym.make('Pendulum-v0')
    state = env.reset()
   
    num_actions = int(env.action_space.high[0] - env.action_space.low[0]) + 1
    obs_dim = env.observation_space.shape[0]
    print("num_actions : %d, obs_dim : %d" % (num_actions, obs_dim))

    action_dim = 1
    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, obs_dim)

    agent = LSPI(num_actions, obs_dim)
    return agent, env, memory, 'Pendulum-v0'

###########################################################################################
def _random_sample(env, memory, agent, game_name, isRender=False):
    state = env.reset()
    csv_name = get_test_record_title(game_name, EPISODE, 'random', num_tests=NUM_EXPERI) + '.csv'

    for i in range(EPISODE):
        env.seed()
        state = env.reset()
        for j in range(TRANSITION):
            if isRender:
                env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                break
        
        # Get Batch of samples & training
        if memory.container_size < BATCH_SIZE:
            sample = memory.select_sample(memory.container_size)
        else:
            sample = memory.select_sample(BATCH_SIZE)
        error = agent.train(sample, lspi_iteration, important_sampling)

        # Get middle mean
        rewards = []
        for j in range(NUM_TESTS_MIDDLE):
            total_reward, _ = test_policy(env, agent)
            rewards.append(total_reward)
        middle = sum(rewards) / NUM_TESTS_MIDDLE
        best = max(rewards)
        worst = min(rewards)

        # Write csv
        with open(csv_name, 'a') as f:
            f.write("{},{},{},{}\n".format(middle, best, worst, error))

        # Print iteration
        if i % 1 == 0:
            print('[episode {}] done'.format(i))

    # for i in range(EPISODE)

    # Write csv (end)
    with open(csv_name, 'a') as f:
        f.write('\n')

    # Clean up
    memory.clear_memory()
   
    ipdb.set_trace()
    return env, agent


def _initial2_sample(env, memory, agent, game_name, isRender=False):
    Best_agent_list = []
    mean_reward = -4000
    Best_mean_reward = -4000
    Best_agent = None
    Best_agent_copy = None
    csv_name = get_test_record_title(game_name, EPISODE, 'initial2', num_tests=NUM_EXPERI) + '.csv'
    bin_name = get_test_record_title(game_name, EPISODE, 'initial2', num_tests=NUM_EXPERI) + '_pickle.bin'

    for i in range(EPISODE):
        # Intializaiton
        env.seed()
        state = env.reset()
        
        # Collect samples 
        for j in range(TRANSITION):
            if isRender:
                env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                break

        # Get batch of samples and Training
        if memory.container_size < BATCH_SIZE: 
            sample = memory.select_sample(memory.container_size)
        else:
            sample = memory.select_sample(BATCH_SIZE)
        error = agent.train(sample, lspi_iteration, important_sampling)
        # Get middle mean
        reward_list = []
        for j in range(NUM_TESTS_MIDDLE):
            total_reward, _ = test_policy(env, agent)
            reward_list.append(total_reward)
            if j % 10 == 0:
                print("test_policy j : {}/{}".format(j, NUM_TESTS_MIDDLE))
        mean_reward = sum(reward_list) / NUM_TESTS_MIDDLE

        # Write csv
        with open(csv_name, 'a') as f:
            f.write("{}\n".format(mean_reward))

        # If getting the best reward, model will collect and learn
                                        # greedy samples at next iteration
        if Best_mean_reward < mean_reward:

            Best_agent = agent
            Best_mean_reward = mean_reward
            memory.clear_memory()
            Best_agent_copy = copy.deepcopy(Best_agent)
            Best_agent_list.append(Best_agent_copy)
            # Write csv
            with open(csv_name, 'a') as f:
                print("# Get Best agent {}".format(mean_reward))
                f.write("# Get Best agent\n")
            
            if os.path.exists(bin_name):
                os.remove(bin_name)
            with open(bin_name, 'wb') as f:
                pickle.dump(Best_agent_list, f) 
        #else 
            # not role back
        

        if i % 10 == 0:
            print('[episode {}] done'.format(i))
    #for i in range(EPISODE)
   
    # Clean up
    memory.clear_memory()

    return env, Best_agent





def _initial_sample(env, memory, agent, game_name, isRender=False):
    Best_agent_list = []
    mean_reward = -4000
    Best_mean_reward = -4000
    Best_agent = None
    Best_agent_copy = None
    csv_name = get_test_record_title(game_name, EPISODE, 'initial', num_tests=NUM_EXPERI) + '.csv'
    bin_name = get_test_record_title(game_name, EPISODE, 'initial', num_tests=NUM_EXPERI) + '_pickle.bin'

    for i in range(EPISODE):
        # Intializaiton
        env.seed()
        state = env.reset()

        # Collect samples 
        for j in range(TRANSITION):
            if isRender:
                env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                break

        # Get batch of samples and Training
        if memory.container_size < BATCH_SIZE: 
            sample = memory.select_sample(memory.container_size)
        else:
            sample = memory.select_sample(BATCH_SIZE)
        error = agent.train(sample, lspi_iteration, important_sampling)

        # Get middle mean
        reward_list = []
        for j in range(NUM_TESTS_MIDDLE):
            total_reward, _ = test_policy(env, agent)
            reward_list.append(total_reward)
        mean_reward = sum(reward_list) / NUM_TESTS_MIDDLE

        # Write csv
        with open(csv_name, 'a') as f:
            f.write("{}\n".format(mean_reward))

        # If getting the best reward, model will collect and learn
                                        # greedy samples at next iteration
        if Best_mean_reward < mean_reward:

            Best_agent = agent
            Best_mean_reward = mean_reward
            memory.clear_memory()
            Best_agent_copy = copy.deepcopy(Best_agent)
            Best_agent_list.append(Best_agent_copy)
            # Write csv
            with open(csv_name, 'a') as f:
                print("# Get Best agent {}".format(mean_reward))
                f.write("# Get Best agent\n")
            
            if os.path.exists(bin_name):
                os.remove(bin_name)
            with open(bin_name, 'wb') as f:
                pickle.dump(Best_agent_list, f)
 
        else:
            # way back
            agent = Best_agent

        if i % 10 == 0:
            print('[episode {}] done'.format(i))
    #for i in range(EPISODE)
   
    # Clean up
    memory.clear_memory()

    return env, Best_agent


def _reuse_sample(env, memory, agent, game_name, isRender=False):
    """ less than 50, collect samples from random, train
        otherwise, collect samples from policy, train
    """
    # Initializaiton
    important_sampling = False
    mean_rewards = []

    for i in range(EPISODE):
        state = env.reset()
        total_reward = 0.0

        # Collect samples
        for j in range(TRANSITION):
            if isRender:
                env.render()
            if i < 50:
                action = env.action_space.sample()
            else:
                action = agent._act(state)
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                break

        # Get batch of samples and Training
        if memory.container_size < BATCH_SIZE:
            sample = memory.select_sample(memory.container_size)
        else:
            sample = memory.select_sample(BATCH_SIZE)
        agent.train(sample, lspi_iteration, important_sampling)

        # Get middle mean
        be_meaned = []
        for j in range(NUM_TESTS_MIDDLE):
            total_reward, _ = test_policy(env, agent)
            be_meaned.append(total_reward)
        middle_mean_reward = sum(be_meaned) / NUM_TESTS_MIDDLE

        # Append middle mean
        mean_rewards.append(total_reward)

        # Print iteration
        if i % 50 == 0:
            print('[episode {}] done'.format(i))

    #end for i in range(EPISODE):

    # Write CSV
    csv_name = get_test_record_title(game_name, EPISODE, 'reuse', num_tests=NUM_EXPERI) + '.csv'
    with open(csv_name, 'a') as f:
        for middle_mean_reward in mean_rewards:
            f.write("{}\n".format(middle_mean_reward))
        f.write('\n')

    # Print mean
    mean = sum(mean_rewards) / EPISODE
    print("_reuse_sample training mean : {}".format(mean))

    # Clean up
    memory.clear_memory()

    return env, agent



def do_experiment(env, memory, agent, game_name, trainopt='random'):
    _sample_and_train = None

    # Training
    if trainopt == 'random':
        env, best_agent = _random_sample(env, memory, agent, game_name, isRender=False)
    elif trainopt == 'initial':
        env, best_agent = _initial_sample(env, memory, agent, game_name, isRender=False)
    elif trainopt == 'initial2':
        env, best_agent = _initial2_sample(env, memory, agent, game_name, isRender=False)
    elif trainopt == 'reuse':
        env, best_agent = _reuse_sample(env, memory, agent, game_name, isRender=False)
    else:
        raise ValueError('wrong trainopt')

    # Testing
    reward_list = []
    for j in range(NUM_TESTS_MIDDLE):
        total_reward, _ = test_policy(env, best_agent)
        reward_list.append(total_reward)
    best_mean_reward = sum(reward_list) / NUM_TESTS_MIDDLE
     
    # Write csv
    csv_name = get_test_record_title(game_name, EPISODE, trainopt, num_tests=NUM_EXPERI) + '.csv'
    with open(csv_name, 'a') as f:
        f.write("best {}\n".format(best_mean_reward))

    return best_agent

def _main(trainopt='initial2'):

    agent, env, memory, game_name = experiment_1()
    #trainopt = 'initial'
    assert trainopt in TRAINOPT

    experiment_title = get_test_record_title(game_name, EPISODE, trainopt, num_tests=NUM_EXPERI)
    print("Experiement title : {}".format(experiment_title))

    csv_name = get_test_record_title(game_name, EPISODE, trainopt, num_tests=NUM_EXPERI) + '.csv'
    print('cvs_file duplication check!')
    if os.path.isfile(csv_name):
        print('Do you want to remove csv_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(csv_name)
        else:
            raise ValueError('you should manually control it!')

    txt_name = get_test_record_title(game_name, EPISODE, trainopt, num_tests=NUM_EXPERI) + '.txt'
    print('txt_file duplication check!')
    if os.path.isfile(txt_name):
        print('Do you want to remove txt_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(txt_name)
        else:
            raise ValueError('you should manually control it!')


    bin_name = get_test_record_title(game_name, EPISODE, 'initial2', num_tests=NUM_EXPERI) + '_pickle.bin'
    print('bin_file duplication check!')
    if os.path.isfile(txt_name):
        print('Do you want to remove bin_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(txt_name)
        else:
            raise ValueError('you should manually control it!')

    for i in range(NUM_EXPERI):
        if i % 10 == 0:
            print("testing ", i)
        env.seed()
        best_agent = do_experiment(env, memory, agent, game_name, trainopt=trainopt)


def pickle_test():

    #env = gym.make('CartPole-v0')
    env = gym.make('Acrobot-v1')

    #bin_name = get_test_record_title('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True) + '_pickle.bin'
    bin_name = get_test_record_title('Acrobot-v1', 1000, 'initial2', num_tests=1, important_sampling=True) + '_pickle.bin'

    #with open('CartPole-v0_EPI1000_initial2_#Test1_important_sampling_pickle.bin', 'rb') as f:
    with open('Acrobot-v1_EPI1000_initial2_#Test1_important_sampling_pickle.bin', 'rb') as f:
        best_agent_list = pickle.load(f)
    
    for i, agent in enumerate(best_agent_list):
        if i != len(best_agent_list)-1:
            continue
        reward_list = []
        for j in range(NUM_TESTS_MIDDLE):
            total_reward, _ = test_policy(env, agent, True)
            reward_list.append(total_reward)
        mean_reward = sum(reward_list) / NUM_TESTS_MIDDLE

        print(mean_reward)



if __name__ == '__main__':
    #_main('initial')
    #_main('initial2')

    pickle_test()

