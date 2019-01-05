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
from test import test_policy

TRANSITION = 15000
EPISODE = 999
BATCH_SIZE = 300
MEMORY_SIZE = TRANSITION + 1000
NUM_TRIALS = 1
NUM_TESTS_PER_EPISODE = 200
TRAINOPT = ['random', 'initial', 'keepBA&notRB', 'reuse']

important_sampling = True
gamma = 0.99

# file name initialization

def acrobot_experiment():
    print('Hello Acrobot world!')
    env = gym.make('Acrobot-v1')

    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    print("state_dim : %d\naction_dim : %d\nnum_actions : %d" % (state_dim, action_dim, num_actions))

    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)

    agent = LSPI(num_actions, state_dim)
    return agent, env, memory, 'Acrobot-v1'

def cartpole_experiment():
    print('Hello CartPole world!')
    env = gym.make('CartPole-v0')

    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    print("state_dim : %d\naction_dim : %d\nnum_actions : %d" % (state_dim, action_dim, num_actions))

    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)

    agent = LSPI(num_actions, state_dim)
    return agent, env, memory, 'CartPole-v0'

def pendulum_experiment():
    exit()
    print('Hello Pendulum world!')
    env = gym.make('Pendulum-v0')

    ipdb.set_trace()
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    ipdb.set_trace()
    action_dim = 1
    print("state_dim : %d\naction_dim : %d\nnum_actions : %d" % (state_dim, action_dim, num_actions))

    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, state_dim)

    agent = LSPI(num_actions, state_dim)
    return agent, env, memory, 'Pendulum-v0'

###########################################################################################
def _random_sample(env, memory, agent, game_name, isRender=False):
    state = env.reset()

    for i in range(EPISODE):
        env.seed()
        state = env.reset()
        for j in range(TRANSITION):
            if isRender:
                env.render()
            # random action sampling
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
        error = agent.train(sample, important_sampling)

        # Get middle mean
        rewards = []
        for j in range(NUM_TESTS_PER_EPISODE):
            total_reward, _ = test_policy(env, agent)
            rewards.append(total_reward)
        middle = sum(rewards) / NUM_TESTS_PER_EPISODE
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
   
    return env, agent


#def _initial2_sample(env, memory, agent, game_name, isRender=False):
def _keep_best_agent_and_not_roll_back_sample(env,
                                              memory,
                                              agent,
                                              game_name,
                                              csv_name,
                                              bin_name,
                                              isRender=False):
    """
    keepBA&notRB
    """
    Best_agent_list = []
    mean_reward = float('-inf')
    Best_mean_reward = float('-inf')
    Best_agent = None
    Best_agent_copy = None

    for i in range(EPISODE):
        # Intializaiton
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
        error = agent.train(sample, important_sampling)

        # Get middle mean
        reward_list = []
        for j in range(NUM_TESTS_PER_EPISODE):
            total_reward, _ = test_policy(env, agent)
            reward_list.append(total_reward)
            # if j % 10 == 0:
            #     print("test_policy j : {}/{}".format(j, NUM_TESTS_PER_EPISODE))
        mean_reward = sum(reward_list) / NUM_TESTS_PER_EPISODE

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

    for i in range(EPISODE):
        # Intializaiton
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
        error = agent.train(sample, important_sampling)

        # Get middle mean
        reward_list = []
        for j in range(NUM_TESTS_PER_EPISODE):
            total_reward, _ = test_policy(env, agent)
            reward_list.append(total_reward)
        mean_reward = sum(reward_list) / NUM_TESTS_PER_EPISODE

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


def _reuse_sample(env,
                  memory,
                  agent,
                  game_name,
                  csv_name,
                  bin_name,
                  isRender=False):
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
                action = agent.act(state)
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
        agent.train(sample, important_sampling)

        # Get middle mean
        be_meaned = []
        for j in range(NUM_TESTS_PER_EPISODE):
            total_reward, _ = test_policy(env, agent)
            be_meaned.append(total_reward)
        middle_mean_reward = sum(be_meaned) / NUM_TESTS_PER_EPISODE

        # Append middle mean
        mean_rewards.append(total_reward)

        # Print iteration
        if i % 50 == 0:
            print('[episode {}] done'.format(i))

    #end for i in range(EPISODE):

    # Write CSV
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

def do_lspi(env, memory, agent, game_name, csv_name, bin_name, trainopt='random'):

    # Training
    if trainopt == 'random':
        env, best_agent = _random_sample(env, memory, agent, game_name, isRender=False)
    elif trainopt == 'initial':
        env, best_agent = _initial_sample(env, memory, agent, game_name, isRender=False)
#    elif trainopt == 'initial2':
#        env, best_agent = _initial2_sample(env, memory, agent, game_name, isRender=False)
    elif trainopt == 'keepBA&notRB':
        env, best_agent = _keep_best_agent_and_not_roll_back_sample(env,
                                                                    memory,
                                                                    agent,
                                                                    game_name,
                                                                    csv_name,
                                                                    bin_name,
                                                                    isRender=False)
    elif trainopt == 'reuse':
        env, best_agent = _reuse_sample(env,
                                        memory,
                                        agent,
                                        game_name,
                                        csv_name,
                                        bin_name,
                                        isRender=False)
    else:
        raise ValueError('wrong trainopt')

    # Testing
    reward_list = []
    for j in range(NUM_TESTS_PER_EPISODE):
        total_reward, _ = test_policy(env, best_agent)
        reward_list.append(total_reward)
    best_mean_reward = sum(reward_list) / NUM_TESTS_PER_EPISODE
     
    # Write csv
    with open(csv_name, 'a') as f:
        f.write("best {}\n".format(best_mean_reward))

    return best_agent

def _main(trainopt='keepBA&notRB', game_title='CartPole'):

    if game_title == "CartPole":
        agent, env, memory, game_name = cartpole_experiment()
    elif game_title == "Acrobot":
        agent, env, memory, game_name = acrobot_experiment()
    elif game_title == "Pendulum":
        agent, env, memory, game_name = pendulum_experiment()
    else:
        print("not implemented yet!")
        exit()

    assert trainopt in TRAINOPT

    experiment_title = get_test_record_title(game_name,
                                             EPISODE,
                                             trainopt,
                                             num_tests=NUM_TRIALS,
                                             important_sampling=important_sampling)
    print("Experiement title : {}".format(experiment_title))
    csv_name = experiment_title + '.csv'
    txt_name = experiment_title + '.txt'
    bin_name = experiment_title + '_pickle.bin'

    print('csv_file duplication check!')
    if os.path.isfile(csv_name):
        print('Do you want to remove csv_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(csv_name)
        else:
            raise ValueError('you should manually control it!')

    print('txt_file duplication check!')
    if os.path.isfile(txt_name):
        print('Do you want to remove txt_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(txt_name)
        else:
            raise ValueError('you should manually control it!')

    print('bin_file duplication check!')
    if os.path.isfile(bin_name):
        print('Do you want to remove bin_file?\n')
        answer = input()
        if answer == 'y' or answer == 'Y':
            os.remove(bin_name)
        else:
            raise ValueError('you should manually control it!')

    for i in range(NUM_TRIALS):
        if i % 1 == 0:
            print("trials ", i, "/", NUM_TRIALS)
        best_agent = do_lspi(env,
                             memory,
                             agent,
                             game_name,
                             csv_name,
                             bin_name,
                             trainopt=trainopt)

    return best_agent

def pickle_test():

    #env = gym.make('CartPole-v0')
    env = gym.make('Acrobot-v1')

    #bin_name = get_test_record_title('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True) + '_pickle.bin'

    #with open('CartPole-v0_EPI1000_initial2_#Test1_important_sampling_pickle.bin', 'rb') as f:
    with open('Acrobot-v1_EPI1000_initial2_#Test1_important_sampling_pickle.bin', 'rb') as f:
        best_agent_list = pickle.load(f)
    
    for i, agent in enumerate(best_agent_list):
        if i != len(best_agent_list)-1:
            continue
        reward_list = []
        for j in range(NUM_TESTS_PER_EPISODE):
            total_reward, _ = test_policy(env, agent, True)
            reward_list.append(total_reward)
        mean_reward = sum(reward_list) / NUM_TESTS_PER_EPISODE

        print(mean_reward)



if __name__ == '__main__':
    #_main('initial')
    _main('keepBA&notRB', 'CartPole')
    #_main('keepBA&notRB', 'Acrobot')
    #_main('keepBA&notRB', 'Pendulum')

    #pickle_test()

