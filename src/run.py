import gym
import numpy as np
import datetime
import ipdb
import os

from replay_memory import Memory
from lspi import LSPI, LSTDQ

#import plot as pl

TRANSITION = 15000
EPISODE = 20
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000
NUM_TESTS = 100
TRAINOPT = ['random', 'initial', 'reuse']

important_sampling = None 
lspi_iteration = 20 #?
num_actions = 3 #?
num_means = 4
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
def _reuse_sample2(env, memory, agent, isRender=False):
    """ less than 50, collect samples from random, train
        otherwise, collect samples from policy, train
    """
    # Initializaiton
    important_sampling = False
    mean_reward = []

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
        total_reward, _ = test_policy(env, agent)

        mean_reward.append(total_reward)
        #if i % 10 == 0:
        #    print('[episode {}] done'.format(i))
    #end for i in range(EPISODE):
    mean = sum(mean_reward) / EPISODE
    print("_reuse_sample2 training mean : {}".format(mean))

    # Clean up
    memory.clear_memory()

    return env, agent

def _initial_sample2(env, memory, agent, game_name, isRender=False):
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

        be_meaned = []
        for j in range(NUM_TESTS):
            total_reward, _ = test_policy(env, agent)
            be_meaned.append(total_reward)
        middle_mean_reward = sum(be_meaned) / NUM_TESTS

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

        mean_rewards.append(middle_mean_reward)
        #if i % 10 == 0:
        #    print('[episode {}] done'.format(i))
    #for i in range(EPISODE)
    
    csv_name = get_test_record_title(game_name, EPISODE, 'initial', num_tests=100) + '.csv'
    with open(csv_name, 'a') as f:
        for middle_mean_reward in mean_rewards:
            f.write("{}\n".format(middle_mean_reward))
        f.write('\n')

    mean = sum(mean_rewards) / EPISODE
    print("_initial_sample2 mean : {}".format(mean))
    
    # Clean up
    memory.clear_memory()

    return env, agent

def _random_sample(env, memory, agent, isRender=False):
    state = env.reset()
    mean_reward = []

    for i in range(EPISODE):
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
        total_reward, _ = test_policy(env, agent)
        mean_reward.append(total_reward)
        #if i % 10 == 0:
        #    print('[episode {}] done'.format(i))
    # for i in range(EPISODE)
    mean = sum(mean_reward) / EPISODE
    print("_random_sample mean : {}".format(mean))

    # Clean up
    memory.clear_memory()
    
    return env, agent

def test_policy(env, agent, isRender=False):
    total_reward = 0.0
    best_policy = 0
    state = env.reset()

    for _ in range(5000):
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

def get_test_record_title(game_name, episode, trainOpt, num_tests=100):
    title = '{}_EPI{}_{}_#Test{}'.format(game_name, episode, trainOpt, num_tests)
    return title

def _test_record(env, agent, trainOpt, episode, game_name, num_tests=100):

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
        f.write('mean : {}\n'.format(mean))
    return mean, txt_name


def do_experiment(env, memory, agent, game_name, trainOpt='random'):
    _sample_and_train = None

    if trainOpt == 'random':
        env, agent = _random_sample(env, memory, agent, isRender=False)
    elif trainOpt == 'initial':
        env, agent = _initial_sample2(env, memory, agent, game_name, isRender=False)
    else:
        raise ValueError('wrong trainOpt')

    mean, txt_name = _test_record(env, agent, trainOpt, EPISODE, game_name, 100)
    print("mean : {}\n".format(mean))

    return mean, txt_name

def main():

    agent, env, memory, game_name = experiment_2()
    #print ("memory size", memory.container_size)

    #y2 = _reuse_sample2(env, memory, agent, name)
    #print("_reuse_sample2 done!")

    trainopt = 'initial'
    assert trainopt in TRAINOPT
    means = []

    csv_name = get_test_record_title(game_name, EPISODE, trainopt, num_tests=100) + '.csv'
    print('cvs_file duplication check!')
    if os.path.isfile(csv_name):
        print('Do you want to remove csv_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(csv_name)
        else:
            raise ValueError('you should manually control it!')

    txt_name = get_test_record_title(game_name, EPISODE, trainopt, num_tests=100) + '.txt'
    print('txt_file duplication check!')
    if os.path.isfile(txt_name):
        print('Do you want to remove txt_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(txt_name)
        else:
            raise ValueError('you should manually control it!')



    for i in range(100):
        if i % 10 == 0:
            print("testing ", i)
        env.seed(i)
        mean, txt_name = do_experiment(env, memory, agent, game_name, trainOpt=trainopt)
        means.append(mean)
    meanmean = sum(means) / 100
    with open(txt_name, 'a') as f:
        f.write("meanmean {}\n".format(meanmean))



    #print("_initial_sample2 done!")

if __name__ == '__main__':
    main()
