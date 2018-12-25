import gym
from record import get_test_record_title
import pickle

def get_best_agent(game_name, episode, trainopt, num_tests=1, important_sampling=True):

    experiment_name = get_test_record_title(game_name, episode, trainopt, num_tests, important_sampling)
    bin_name = experiment_name + '_pickle.bin'
    print("Reading {}...".format(bin_name))

    with open(bin_name, 'rb') as f:
        best_agent_list = pickle.load(f)
    best_agent = best_agent_list[-1]

    return best_agent

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

def test_agent(game_name, agent, num_iteration, isRender=False): 

    env = gym.make(game_name)

    reward_list = []
    for j in range(num_iteration):
        total_reward, _ = test_policy(env, agent, isRender=isRender)
        reward_list.append(total_reward)
    mean_reward = sum(reward_list) / num_iteration

    print("[test results] : {} (num_iteration : {})".format(mean_reward, num_iteration))
    return mean_reward

if __name__ == "__main__":
    NUM_TEST_ITER = 100
    best_agent = get_best_agent('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True)
    mean_reward = test_agent('CartPole-v0', best_agent, NUM_TEST_ITER, isRender=True)

