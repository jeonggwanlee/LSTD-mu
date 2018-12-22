import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import ipdb

from replay_memory import Memory
from lspi import LSPI, LSTDQ
#import plot as pl

TRANSITION = 15000
EPISODE = 100
BATCH_SIZE = 400
MEMORY_SIZE = TRANSITION + 1000

important_sampling = None 
lspi_iteration = 20 #?
num_actions = 3 #?
num_means = 4
gamma = 0.99

mean_reward1 = []
mean_reward2 = []

def experiment_1():
    print('Hello Acrobot world!')
    env = gym.make('Acrobot-v1')
    state = env.reset()

    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    print("num_actions : %d, obs_dim : %d" % (num_actions, obs_dim))
    
    action_dim = 1
    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, obs_dim)

    agent = LSPI(num_actions, obs_dim)
    return agent, env, memory, 'Acrobot-v1'

def experiment_2():
    print('Hello CartPole world!')
    env = gym.make('CartPole-v0')
    state = env.reset()

    num_actions = env.action_space.n + 1
    obs_dim = env.observation_space.shape[0]
    print("num_actions : %d, obs_dim : %d" % (num_actions, obs_dim))

    action_dim = 1
    memory = Memory(MEMORY_SIZE, BATCH_SIZE, action_dim, obs_dim)

    agent = LSPI(num_actions, obs_dim)
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


def _reuse_sample2(env, memory, agent, name):
    """
    memory. Memory.
    agent. LSPI.
    """

    state = env.reset()
    total_reward = 0.0
    important_sampling = False

    for j in range(EPISODE):
        state = env.reset()
        total_reward = 0.0

        for i in range(TRANSITION):
            #env.render()
            if j < 50:
                action = env.action_space.sample()
            else:
                action = agent._act(state)
                sample = memory.select_sample(BATCH_SIZE)
                agent.train(sample, lspi_iteration, important_sampling)
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                print ("Get the goal(done)")
                break
        #end for i in range(TRANSITION):
        if i < BATCH_SIZE:
            sample = memory.select_sample(i)
        else:
            sample = memory.select_sample(BATCH_SIZE)

        agent.train(sample, lspi_iteration, important_sampling)
        total_reward, policy_test = test_policy(env, state, agent)
        mean_reward2.append(total_reward)

        with open("{}.csv".format(name), 'a') as f:
            f.write("{},{}\n".format(j, total_reward))
        print("[episode {}/{}] total_reward : {}".format(j, EPISODE, total_reward))
        #print("memory(deque).container_size : ", memory.container_size)
    #end for j in range(EPISODE):

    ipdb.set_trace()
    while 1:
        total_reward, policy_test = test_policy(env, state, agent, render=True)
        print (total_reward) 


    return mean_reward2


def test_policy(env, state, agent, render=False):

    total_reward = 0.0
    Best_policy=0
    state = env.reset()

    for i in range(5000):
        if render:
            env.render()
        action=agent._act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

        total_reward += gamma * reward
        if done:
            Best_policy=agent.policy
            break

    return total_reward, Best_policy


def _initial_sample2(env, memory, agent, name):
    state = env.reset()
    total_reward = -4000
    best_reward = -4000
    Best_agent = None
    best_theta = False
    found = False


    for j in range(EPISODE):
        state = env.reset()
        best_theta = False

        for i in range(TRANSITION):
            env.render()
            #if best_reward >= total_reward: # 베스트리워드가 토탈 리워드 보다 크거나 같을 때
            if not found:
                action = env.action_space.sample()
            else:
                agent=Best_agent    
                best_theta = True   
                action = agent._act(state)
            next_state, reward, done, info = env.step(action)      
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                print("done iteration = %d" % (i))
                break
        #end for i in range(TRANSITION)

        if done: # enough, so small train
            sample = memory.select_sample(i)
        else: # If done is False, it means lack of training?
            sample = memory.select_sample(TRANSITION)
        
        agent.train(sample, lspi_iteration, important_sampling)
        total_reward, policy_test = test_policy(env, state, agent)

        if best_reward < total_reward:
            Best_agent = agent
            best_reward = total_reward
            #total_reward = -4950.0
            found = True
        else:
            found = False

        with open("_initial_sample2_{}.csv".format(name), 'a') as f:
            f.write("{},{}\n".format(j, total_reward))
        print("_initial_sample2 [episode {}/{}] total_reward : {}".format(j, EPISODE, total_reward))

        if best_theta:
            memory.clear_memory()
        mean_reward1.append(total_reward)
    #for j in range(EPISODE)
    memory.clear_memory()
    
    ipdb.set_trace()
    while 1:
        total_reward, policy_test = test_policy(env, state, agent)
        print (total_reward) 

    return mean_reward1

def main():

    agent, env, memory, name = experiment_2()
    #print ("memory size", memory.container_size)

    y2 = _reuse_sample2(env, memory, agent, name)
    #print("_reuse_sample2 done!")
    #y1 = _initial_sample2(env, memory, agent, name)
    #print("_initial_sample2 done!")

    #x = np.arange(0, len(mean_reward1))

    #np.reshape(mean_reward1, x.shape)
    #print (x.shape, mean_reward1, mean_reward2, x)

    #ipdb.set_trace()
    #pj = pl.Plot()
    #pj.plot_rewad(x, y1, y2)

if __name__ == '__main__':
    main()
