import gym
import numpy as np
import datetime
import ipdb
import os

from replay_memory import Memory
from lspi import LSPI, LSTDQ

#import plot as pl

TRANSITION = 15000
EPISODE = 1000
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000
NUM_EXPERI = 1
NUM_TESTS_MIDDLE = 1000
TRAINOPT = ['random', 'initial', 'reuse']

important_sampling = None 
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
        agent.train(sample, lspi_iteration, important_sampling)

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
            f.write("{},{},{}\n".format(middle, best, worst))

        # Print iteration
        if i % 10 == 0:
            print('[episode {}] done'.format(i))

    # for i in range(EPISODE)

    # Write csv (end)
    with open(csv_name, 'a') as f:
        f.write('\n')

    # Clean up
    memory.clear_memory()
   
    ipdb.set_trace()
    return env, agent


def _initial_sample(env, memory, agent, game_name, isRender=False):
    total_reward = -4000
    Best_reward = -4000
    Best_agent = None
    Best_theta = False
    Found_best = False
    mean_rewards = []

    for i in range(EPISODE):
        # Intializaiton
        state = env.reset()
        Best_theta = False

        # Collect samples 
        for j in range(TRANSITION):
            if isRender:
                env.render()
            if Found_best is False:
                action = env.action_space.sample()
            else:
                agent = Best_agent    
                Best_theta = True   
                action = agent._act(state)
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                done_iter = j
                break

        # Get batch of samples and Training
        if done: 
            sample = memory.select_sample(done_iter)
        else:
            sample = memory.select_sample(BATCH_SIZE)
        agent.train(sample, lspi_iteration, important_sampling)

        # Get middle mean
        be_meaned = []
        for j in range(NUM_TESTS_MIDDLE):
            total_reward, _ = test_policy(env, agent)
            be_meaned.append(total_reward)
        middle_mean_reward = sum(be_meaned) / NUM_TESTS_MIDDLE

        # If getting the best reward, model will collect and learn
                                        # greedy samples at next iteration
        if Best_reward < total_reward:
            Best_agent = agent
            Best_reward = total_reward
            Found_best = True
        else:
            Found_best = False

        # After learning greedy samples, reinitialize memory
        if Best_theta:
            memory.clear_memory()

        # Append middle mean
        mean_rewards.append(middle_mean_reward)
        if i % 50 == 0:
            print('[episode {}] done'.format(i))
    #for i in range(EPISODE)
   
    # Write csv
    csv_name = get_test_record_title(game_name, EPISODE, 'initial', num_tests=NUM_EXPERI) + '.csv'
    with open(csv_name, 'a') as f:
        for middle_mean_reward in mean_rewards:
            f.write("{}\n".format(middle_mean_reward))
        f.write('\n')

    # Print mean
    mean = sum(mean_rewards) / EPISODE
    print("_initial_sample training mean : {}".format(mean))
    
    # Clean up
    memory.clear_memory()

    return env, agent


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


def test_policy(env, agent, isRender=False):
    total_reward = 0.0
    best_policy = 0
    env.seed()

    state = env.reset()
    for i in range(5000):
        if isRender:
            env.render()
        action = agent._act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

        total_reward += reward
        if done:
            best_policy = agent.policy
            break

    return total_reward, best_policy

def get_test_record_title(game_name, episode, trainOpt, num_tests=NUM_EXPERI):
    title = '{}_EPI{}_{}_#Test{}'.format(game_name, episode, trainOpt, num_tests)
    return title

def _test_record(env, agent, trainOpt, episode, game_name, num_tests=NUM_EXPERI):
    """DEPRECATED
    """

    be_meaned = []
    txt_name = get_test_record_title(game_name, episode, trainOpt, num_tests) + '.txt'

    #print('...recoding({})...'.format(txt_name))
    with open(txt_name, 'a') as f:
        f.write(datetime.datetime.now().strftime('Record time : %Y-%m-%d_%H-%M-%S\n'))
        for i in range(num_tests):
            total_reward, _ = test_policy(env, agent)
            #f.write('{}\n'.format(total_reward))
            be_meaned.append(total_reward)
        mean = sum(be_meaned) / num_tests
        f.write('testing mean : {}\n'.format(mean))
    return mean, txt_name


def do_experiment(env, memory, agent, game_name, trainOpt='random'):
    _sample_and_train = None

    # Training
    if trainOpt == 'random':
        env, agent = _random_sample(env, memory, agent, game_name, isRender=False)
    elif trainOpt == 'initial':
        env, agent = _initial_sample(env, memory, agent, game_name, isRender=False)
    elif trainOpt == 'reuse':
        env, agent = _reuse_sample(env, memory, agent, game_name, isRender=False)
    else:
        raise ValueError('wrong trainOpt')

    # Testing
#    mean, txt_name = _test_record(env, agent, trainOpt, EPISODE, game_name, NUM_EXPERI)
#    print("testing mean : {}\n".format(mean))
##    txt_name = get_test_record_title(game_name, EPISODE, trainOpt, NUM_EXPERI) + '.txt'
##    mean = 0
##    return mean, txt_name
    return

def _main(trainopt='initial'):

    agent, env, memory, game_name = experiment_2()
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

    #means = []
    for i in range(NUM_EXPERI):
        if i % 10 == 0:
            print("testing ", i)
        env.seed(i)
        #mean, txt_name = do_experiment(env, memory, agent, game_name, trainOpt=trainopt)
        do_experiment(env, memory, agent, game_name, trainOpt=trainopt)
        #means.append(mean)
    #meanmean = sum(means) / NUM_EXPERI
    #with open(txt_name, 'a') as f:
    #    f.write("meanmean {}\n".format(meanmean))


if __name__ == '__main__':
    _main('random')
    #_main('initial')
    #_main('reuse')
