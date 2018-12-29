import pickle
import gym
import ipdb
NUM_EVALUATION = 100
TRANSITION = 3000
isRender = True

best_policy_bin_name = "CartPole-v0_statedim4_numbasis10_best_policy_pickle.bin"

with open(best_policy_bin_name, 'rb') as f:
    best_policy, t_collection = pickle.load(f)

best_agent = best_policy[-1]
env = gym.make('CartPole-v0')

for iter, best_agent in enumerate(best_policy):
    if iter != 47:
        continue
    total_rewards = []
    for i in range(NUM_EVALUATION):
        env.seed()
        state = env.reset()
        total_reward = 0.0

        for tr in range(TRANSITION):
            if isRender:
                env.render()
            action = best_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                print(tr)


        # for _
        ipdb.set_trace()
        print(total_reward)
        total_rewards.append(total_reward)

    reward = sum(total_rewards) / NUM_EVALUATION
    print("{}'s agent {} ".format(iter, reward))
