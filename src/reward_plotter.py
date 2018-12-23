import ipdb
import numpy as np
EPISODE = 20
NUM_TESTS = 100
TRAINOPT = ['random', 'initial']
trainopt = 'initial'
assert trainopt in TRAINOPT
game_name = 'CartPole-v0'

print("What is the episode size?(default==20)")
ep = input()
EPISODE = int(ep)
print("Episode is ", EPISODE)

def get_test_record_title(game_name, episode, trainOpt, num_tests=100):
    title = '{}_EPI{}_{}_#Test{}'.format(game_name, episode, trainOpt, num_tests)
    return title

csv_name= get_test_record_title(game_name, EPISODE, trainopt, num_tests=NUM_TESTS) + '.csv'

with open(csv_name, 'r') as rf:
    lines = [line for line in rf]
    rewards = []
    reward = []
    isNew = False
    for l in lines:
        if l == '\n':
            rewards.append(reward)
            isNew = True
            continue
        if isNew:
            reward = []
            isNew = False
            reward.append(float(l))
        else:
            reward.append(float(l))
        
    rewards = np.asarray(rewards)
    rewardsT = rewards

new_csv_name = get_test_record_title(game_name, EPISODE, trainopt, num_tests=NUM_TESTS) + '_plot2.csv'


with open(new_csv_name, 'w') as wf:
    
    for i in range(rewardsT.shape[0]):
        for j in range(rewardsT.shape[1]):
            if j == 0 or j == rewardsT.shape[1]-1:
                wf.write("{}".format(rewardsT[i][j]))
            #if j != rewardsT.shape[1]-1:
            if j == 0:
                wf.write(",")
        wf.write('\n')
