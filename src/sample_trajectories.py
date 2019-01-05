import pickle
import gym
import ipdb

from test import get_best_agent, test_policy
from record import get_test_record_title

TRANSITION = 15000
NUM_TRAJECTORIES = 100

def sample_trajectories_from_expert(game_name,
                                    episode,
                                    trainopt,
                                    num_tests=1,
                                    important_sampling=True,
                                    num_trajectories=100):
    best_agent = get_best_agent(game_name,
                                episode,
                                trainopt,
                                num_tests,
                                important_sampling)
    env = gym.make(game_name)

    trajectories = []
    for i in range(num_trajectories):
        state = env.reset()
        trajectory = []
        for j in range(TRANSITION):
            action = best_agent.act(state)
            next_state, reward, done, info = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            state = next_state
            if done:
                if i % 10 == 0:
                    print("trajectories done j {}/{}".format(j, num_trajectories))
                break

        trajectories.append(trajectory)

    experiment_name = get_test_record_title(game_name, episode, trainopt, num_tests, important_sampling)
    trajectory_name = "{}_#Trajectories{}_pickle.bin".format(experiment_name, num_trajectories)
    ipdb.set_trace()

    with open(trajectory_name, 'wb') as f:
        pickle.dump(trajectories, f)

if __name__ == "__main__":
    # sample_trajectories_from_expert('CartPole-v0',
    #                                1000,
    #                                'initial2',
    #                                num_tests=1,
    #                                important_sampling=True,
    #                                num_trajectories=100)

    sample_trajectories_from_expert('CartPole-v0',
                                    999,
                                    'keepBA&notRB',
                                    num_tests=1,
                                    important_sampling=True,
                                    num_trajectories=NUM_TRAJECTORIES)
