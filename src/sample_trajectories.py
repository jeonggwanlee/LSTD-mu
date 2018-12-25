import pickle
import gym

from test import get_best_agent, test_policy
from record import get_test_record_title

TRANSITION = 15000

def sample_trajectories_from_expert(game_name, episode, trainopt, num_tests=1, important_sampling=True, num_traj=100):
    best_agent = get_best_agent(game_name, episode, trainopt, num_tests, important_sampling)

    env = gym.make(game_name)

    trajectories = []
    for i in range(num_traj):
        env.seed()
        state = env.reset()
        
        trajectory = []
        for j in range(TRANSITION):
            
            action = best_agent._act(state)
            next_state, reward, done, info = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            state = next_state
            if done:
                if i % 10 == 0:
                    print("done j {}".format(j))
                break

        trajectories.append(trajectory)

    experiment_name = get_test_record_title(game_name, episode, trainopt, num_tests, important_sampling)
    trajectory_name = "{}_num_traj{}_pickle.bin".format(experiment_name, num_traj)
    
    with open(trajectory_name, 'wb') as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    sample_trajectories_from_expert('CartPole-v0', 1000, 'initial2', num_tests=1, important_sampling=True,
                                    num_traj=100)
